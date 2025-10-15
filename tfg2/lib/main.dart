import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' show MediaType;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TFG2',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const RecorderPage(),
    );
  }
}

class RecorderPage extends StatefulWidget {
  const RecorderPage({super.key});
  @override
  State<RecorderPage> createState() => _RecorderPageState();
}

class _AudioFile {
  final String path;
  final int size;
  final DateTime modified;
  _AudioFile({required this.path, required this.size, required this.modified});
  String get name => path.split(Platform.pathSeparator).last;
  String get sizeKB => '${(size / 1024).toStringAsFixed(1)} KB';
}

class _RecorderPageState extends State<RecorderPage> {
  final _recorder = AudioRecorder();
  final _player = AudioPlayer();
  bool _isRecording = false;
  String _status = 'Listo';
  List<_AudioFile> _audios = [];
  String? _playingPath;

  // Carpeta: /storage/emulated/0/Android/data/<package>/files/audios
  Future<Directory> _audioDir() async {
    final ext = await getExternalStorageDirectory(); // app-specific externo
    final dir = Directory('${ext!.path}/audios');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  Future<String> _nextAudioPath() async {
    final dir = await _audioDir();
    final ts = DateTime.now().millisecondsSinceEpoch;
    return '${dir.path}/rec_$ts.wav';
  }

  Future<void> _ensureMic() async {
    final mic = await Permission.microphone.request();
    if (!mic.isGranted) throw Exception('Permiso de micrófono denegado');
  }

  Future<void> _start() async {
    try {
      await _ensureMic();
      final path = await _nextAudioPath();
      final config = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      );
      await _recorder.start(config, path: path);
      setState(() {
        _isRecording = true;
        _status = 'Grabando…';
      });
    } catch (e) {
      setState(() => _status = 'Error al iniciar: $e');
    }
  }

  Future<void> _stop() async {
    try {
      await _recorder.stop(); // el path ya lo guardamos al iniciar
      setState(() {
        _isRecording = false;
        _status = 'Grabación guardada';
      });
      await _refreshList();
    } catch (e) {
      setState(() => _status = 'Error al parar: $e');
    }
  }

  Future<void> _refreshList() async {
    final dir = await _audioDir();
    final files = await dir
        .list()
        .where((e) => e is File && e.path.toLowerCase().endsWith('.wav'))
        .cast<File>()
        .toList();

    final list = <_AudioFile>[];
    for (final f in files) {
      final st = await f.stat();
      list.add(_AudioFile(path: f.path, size: st.size, modified: st.modified));
    }
    list.sort((a, b) => b.modified.compareTo(a.modified)); // recientes primero
    setState(() => _audios = list);
  }

  Future<void> _play(String path) async {
    try {
      await _player.stop();
      await _player.play(DeviceFileSource(path));
      setState(() => _playingPath = path);
      _player.onPlayerComplete.listen((_) {
        if (mounted) setState(() => _playingPath = null);
      });
    } catch (e) {
      setState(() => _status = 'No se pudo reproducir: $e');
    }
  }

  Future<void> _pause() async {
    await _player.pause();
    setState(() => _playingPath = null);
  }

  Future<void> _delete(String path) async {
    try {
      if (_playingPath == path) {
        await _player.stop();
        _playingPath = null;
      }
      await File(path).delete();
      await _refreshList();
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Borrado: ${path.split('/').last}')));
      }
    } catch (e) {
      setState(() => _status = 'No se pudo borrar: $e');
    }
  }

  Future<void> _sendToServer(String path) async {
    if (!File(path).existsSync()) {
      setState(() => _status = 'Archivo no encontrado');
      return;
    }
    try {
      setState(() => _status = 'Enviando a Render…');
      final uri = Uri.parse('https://tfg-server.onrender.com/transcribe'); 
      final req = http.MultipartRequest('POST', uri);
      req.files.add(await http.MultipartFile.fromPath(
        'file',
        path,
        contentType: MediaType('audio', 'wav'),
      ));
      final streamed = await req.send();
      final res = await http.Response.fromStream(streamed);
      if (res.statusCode == 200) {
        final data = json.decode(res.body) as Map<String, dynamic>;
        // Muestra resumen rápido
        final tempo = data['tempo'];
        final ts = data['time_signature'];
        final count = (data['notes'] as List?)?.length ?? 0;
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('OK: tempo $tempo · compás $ts · $count notas')),
          );
        }
        setState(() => _status = 'Procesado');
      } else {
        setState(() => _status = 'Error ${res.statusCode}: ${res.body}');
      }
    } catch (e) {
      setState(() => _status = 'Fallo al enviar: $e');
    }
  }

  @override
  void initState() {
    super.initState();
    _refreshList();
  }

  @override
  void dispose() {
    _player.dispose();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('TFG2 – Grabador & IA')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(_status),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isRecording ? null : _start,
                    icon: const Icon(Icons.mic),
                    label: const Text('Grabar'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isRecording ? _stop : null,
                    icon: const Icon(Icons.stop),
                    label: const Text('Parar'),
                  ),
                ),
              ],
            ),
            const Divider(height: 32),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Grabaciones', style: TextStyle(fontWeight: FontWeight.bold)),
                IconButton(
                  tooltip: 'Actualizar',
                  onPressed: _refreshList,
                  icon: const Icon(Icons.refresh),
                ),
              ],
            ),
            Expanded(
              child: _audios.isEmpty
                  ? const Center(child: Text('No hay grabaciones aún'))
                  : ListView.separated(
                      itemCount: _audios.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final f = _audios[i];
                        final isPlaying = _playingPath == f.path;
                        return ListTile(
                          leading: IconButton(
                            icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                            onPressed: isPlaying ? _pause : () => _play(f.path),
                          ),
                          title: Text(f.name, maxLines: 1, overflow: TextOverflow.ellipsis),
                          subtitle: Text(
                              '${f.sizeKB} · ${f.modified.toLocal().toString().split(".").first}'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                tooltip: 'Enviar a Render',
                                icon: const Icon(Icons.cloud_upload),
                                onPressed: () => _sendToServer(f.path),
                              ),
                              IconButton(
                                tooltip: 'Borrar',
                                icon: const Icon(Icons.delete_outline),
                                onPressed: () => _delete(f.path),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
