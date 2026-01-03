import 'dart:async';
// Timer → untuk menjalankan deteksi otomatis tiap 500 ms

import 'dart:convert';
// jsonEncode / jsonDecode → kirim & terima data JSON ke backend

import 'package:flutter/foundation.dart';
// kIsWeb → cek apakah aplikasi dijalankan di Web atau Mobile

import 'package:flutter/material.dart';
// Material UI Flutter

import 'package:camera/camera.dart';
// Plugin kamera → ambil gambar dari kamera HP / webcam

import 'package:http/http.dart' as http;
// HTTP client → kirim gambar ke backend Flask

import 'package:path/path.dart' as path;
// Ambil nama file dari path gambar

import 'package:mime/mime.dart';
// Menentukan tipe file (image/jpeg, dll)

import 'package:http_parser/http_parser.dart';
// Parsing multipart upload (untuk Android/iOS)

import 'package:fl_chart/fl_chart.dart';
// Library grafik → tampilkan movement pupil

import 'pages/hasil_page.dart';
// Halaman hasil PMB

// =========================================================
// MAIN — TITIK AWAL APLIKASI
// (PERBAIKAN HANYA DI BAGIAN KAMERA: pilih kamera depan)
// =========================================================
Future<void> main() async {
  // Pastikan binding Flutter siap sebelum pakai kamera
  WidgetsFlutterBinding.ensureInitialized();

  // Ambil daftar kamera yang tersedia di device
  final cameras = await availableCameras();

  // =========================================================
  // PERBAIKAN KAMERA:
  // - Jangan pakai cameras.first (kadang itu kamera belakang)
  // - Pilih kamera depan berdasarkan lensDirection
  // - Kalau tidak ketemu (device aneh), fallback ke camera pertama
  // =========================================================
  final CameraDescription selectedCamera = cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.front,
    orElse: () => cameras.first,
  );

  // Jalankan aplikasi
  runApp(MataApp(camera: selectedCamera));
}

// =========================================================
// ROOT APP — PEMBUNGKUS APLIKASI
// =========================================================
class MataApp extends StatelessWidget {
  final CameraDescription camera;

  const MataApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // hilangkan banner debug
      title: 'Eye Screening PMB',
      home: MataFormPage(camera: camera), // halaman utama
    );
  }
}

// =========================================================
// HALAMAN UTAMA APLIKASI
// =========================================================
class MataFormPage extends StatefulWidget {
  final CameraDescription camera;

  const MataFormPage({super.key, required this.camera});

  @override
  State<MataFormPage> createState() => _MataFormPageState();
}

class _MataFormPageState extends State<MataFormPage> {
  // Controller kamera
  late CameraController _controller;

  // Future untuk nunggu kamera siap
  late Future<void> _initializeControllerFuture;

  // Timer untuk auto-detect
  Timer? _timer;

  // Flag supaya tidak double request
  bool _isDetecting = false;

  // Teks hasil deteksi dari backend
  String? hasilDeteksi;

  // =========================================================
  // TITIK PUPIL DI KAMERA (DEMO VISUAL)
  // =========================================================

  bool showPupilOnCamera = false;
  // true  → tampilkan titik pupil di kamera
  // false → sembunyikan

  Offset? leftPupilPx;
  Offset? rightPupilPx;
  // Posisi pupil dalam pixel (dikirim backend)

  // =========================================================
  // GRAFIK MOVEMENT PUPIL
  // =========================================================

  List<double> movementLeftHistory = [];
  List<double> movementRightHistory = [];
  // Menyimpan history movement untuk grafik

  int maxPoints = 20;
  // Maksimal titik di grafik supaya tidak kepanjangan

  // =========================================================
  // FORM PMB
  // =========================================================

  final _formKey = GlobalKey<FormState>();

  final namaCtrl = TextEditingController();
  final emailCtrl = TextEditingController();
  final alamatCtrl = TextEditingController();
  final pekerjaanCtrl = TextEditingController();
  final waCtrl = TextEditingController();
  final sekolahCtrl = TextEditingController();

  // =========================================================
  // ALAMAT BACKEND
  // =========================================================
  String baseUrl = "http://192.168.0.107:5000/";

  // =========================================================
  // INITSTATE — DIJALANKAN SAAT HALAMAN DIBUKA
  // =========================================================
  @override
  void initState() {
    super.initState();

    // Inisialisasi kamera
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium, // kualitas sedang (stabil & ringan)
      enableAudio: false,
    );

    // Tunggu kamera siap, lalu mulai auto-detect
    _initializeControllerFuture = _controller.initialize();
    _initializeControllerFuture.then((_) => _startAutoDetect());
  }

  @override
  void dispose() {
    // Hentikan timer & kamera saat halaman ditutup
    _timer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  // =========================================================
  // AUTO DETECT — JALAN SETIAP 500 ms
  // =========================================================
  void _startAutoDetect() {
    _timer = Timer.periodic(
      const Duration(milliseconds: 500),
      (_) {
        if (!_isDetecting) _deteksiMata();
      },
    );
  }

  // =========================================================
  // AMBIL FOTO + KIRIM KE BACKEND
  // =========================================================
  Future<void> _deteksiMata() async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      // Ambil gambar dari kamera
      final XFile picture = await _controller.takePicture();
      final bytes = await picture.readAsBytes();

      final uri = Uri.parse("$baseUrl/detect");
      dynamic responseData;

      // ===== MODE WEB =====
      if (kIsWeb) {
        final payload = jsonEncode({
          "image": "data:image/jpeg;base64,${base64Encode(bytes)}",
        });

        final response = await http.post(
          uri,
          headers: {"Content-Type": "application/json"},
          body: payload,
        );

        responseData = jsonDecode(response.body);

        // ===== MODE ANDROID / IOS =====
      } else {
        final mime = lookupMimeType(picture.path) ?? "image/jpeg";

        final request = http.MultipartRequest("POST", uri);
        request.files.add(
          http.MultipartFile.fromBytes(
            'image',
            bytes,
            filename: path.basename(picture.path),
            contentType: MediaType.parse(mime),
          ),
        );

        final streamed = await request.send();
        final respStr = await streamed.stream.bytesToString();
        responseData = jsonDecode(respStr);
      }

      // Jika pupil tidak terdeteksi
      if (responseData["status"] == "pupil_not_found") {
        setState(() {
          hasilDeteksi = "Pupil tidak ditemukan";
          leftPupilPx = null;
          rightPupilPx = null;
        });
      } else {
        _prosesHasil(responseData);
      }
    } catch (e) {
      setState(() {
        hasilDeteksi = "Error: $e";
      });
    } finally {
      _isDetecting = false;
    }
  }

  // =========================================================
  // PROSES DATA DARI BACKEND
  // =========================================================
  void _prosesHasil(dynamic data) {
    final left = data["pupil"]["left"];
    final right = data["pupil"]["right"];

    // Ambil nilai movement pupil
    final movementLeft = (left["movement_norm"] as num).toDouble();
    final movementRight = (right["movement_norm"] as num).toDouble();

    // Ambil posisi pupil untuk overlay kamera
    leftPupilPx = (left["center_x"] != null && left["center_y"] != null)
        ? Offset(
            (left["center_x"] as num).toDouble(),
            (left["center_y"] as num).toDouble(),
          )
        : null;

    rightPupilPx = (right["center_x"] != null && right["center_y"] != null)
        ? Offset(
            (right["center_x"] as num).toDouble(),
            (right["center_y"] as num).toDouble(),
          )
        : null;

    // Simpan ke grafik
    movementLeftHistory.add(movementLeft);
    movementRightHistory.add(movementRight);

    // Batasi jumlah titik grafik
    if (movementLeftHistory.length > maxPoints) {
      movementLeftHistory.removeAt(0);
      movementRightHistory.removeAt(0);
    }

    // Teks hasil
    hasilDeteksi =
        "${data["status"]}\nMovement L: ${movementLeft.toStringAsFixed(4)} | R: ${movementRight.toStringAsFixed(4)}";

    setState(() {});
  }

  // =========================================================
  // GRAFIK MOVEMENT PUPIL
  // =========================================================
  Widget _buildMovementChart() {
    return SizedBox(
      height: 250,
      child: LineChart(
        LineChartData(
          titlesData: const FlTitlesData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: List.generate(
                movementLeftHistory.length,
                (i) => FlSpot(i.toDouble(), movementLeftHistory[i]),
              ),
              isCurved: true,
              color: Colors.blue,
              barWidth: 2,
              dotData: const FlDotData(show: false),
            ),
            LineChartBarData(
              spots: List.generate(
                movementRightHistory.length,
                (i) => FlSpot(i.toDouble(), movementRightHistory[i]),
              ),
              isCurved: true,
              color: Colors.red,
              barWidth: 2,
              dotData: const FlDotData(show: false),
            ),
          ],
        ),
      ),
    );
  }

  // =========================================================
  // CAMERA + OVERLAY TITIK PUPIL
  // =========================================================
  Widget _buildCameraWithOverlay() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final previewSize = _controller.value.previewSize;

        // Kalau kamera belum siap
        if (previewSize == null) {
          return CameraPreview(_controller);
        }

        final boxW = constraints.maxWidth;
        final boxH = boxW * (previewSize.height / previewSize.width);

        Widget overlayDot(Offset? pupilPx, Color color) {
          if (!showPupilOnCamera || pupilPx == null) {
            return const SizedBox.shrink();
          }

          final dx = (pupilPx.dx / previewSize.width) * boxW;
          final dy = (pupilPx.dy / previewSize.height) * boxH;

          return Positioned(
            left: dx - 6,
            top: dy - 6,
            child: Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
            ),
          );
        }

        return SizedBox(
          width: boxW,
          height: boxH,
          child: Stack(
            children: [
              Positioned.fill(child: CameraPreview(_controller)),
              overlayDot(leftPupilPx, Colors.blue),
              overlayDot(rightPupilPx, Colors.red),
            ],
          ),
        );
      },
    );
  }

  // =========================================================
  // UI UTAMA
  // =========================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFE7F3FA),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              const Text(
                "Screening Mata & Form PMB",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              FutureBuilder(
                future: _initializeControllerFuture,
                builder: (_, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    return _buildCameraWithOverlay();
                  }
                  return const CircularProgressIndicator();
                },
              ),
              const SizedBox(height: 16),
              Text(
                hasilDeteksi ?? "Mendeteksi...",
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
              // Toggle demo titik pupil
              SwitchListTile(
                title: const Text("Tampilkan titik pupil di kamera (demo)"),
                value: showPupilOnCamera,
                onChanged: (v) => setState(() => showPupilOnCamera = v),
              ),
              const SizedBox(height: 20),
              if (movementLeftHistory.isNotEmpty) _buildMovementChart(),
              const SizedBox(height: 20),
              _buildForm(),
            ],
          ),
        ),
      ),
    );
  }

  // =========================================================
  // FORM PMB
  // =========================================================
  Widget _buildForm() {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          _field("Nama Lengkap", namaCtrl),
          _field("Email", emailCtrl),
          _field("Alamat", alamatCtrl),
          _field("Pekerjaan", pekerjaanCtrl),
          _field("No WA", waCtrl),
          _field("Asal Sekolah", sekolahCtrl),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () {
              if (_formKey.currentState!.validate()) {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => HasilPage(
                      nama: namaCtrl.text,
                      email: emailCtrl.text,
                      alamat: alamatCtrl.text,
                      pekerjaan: pekerjaanCtrl.text,
                      wa: waCtrl.text,
                      sekolah: sekolahCtrl.text,
                      tglLahir: "",
                      jenisKelamin: "Perempuan",
                      prodi: "Teknik Informatika",
                      hasilDeteksi: hasilDeteksi ?? "-",
                    ),
                  ),
                );
              }
            },
            child: const Text("Kirim"),
          ),
        ],
      ),
    );
  }

  Widget _field(String label, TextEditingController c) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        TextFormField(
          controller: c,
          validator: (v) => v == null || v.isEmpty ? "Harus diisi" : null,
        ),
        const SizedBox(height: 10),
      ],
    );
  }
}
