import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'package:camera/camera.dart';               // untuk kamera HP
import 'package:http/http.dart' as http;           // untuk kirim foto ke backend
import 'package:path/path.dart' as path;           // untuk mendapat nama file
import 'package:mime/mime.dart';                   // mengetahui jenis file
import 'package:http_parser/http_parser.dart';     // parser multipart upload
import 'package:fl_chart/fl_chart.dart';           // untuk grafik movement pupil

import 'pages/hasil_page.dart';                    // halaman hasil akhir


// =========================================================
//  FUNGSI MAIN – aplikasi dimulai dari sini
// =========================================================
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // ambil daftar kamera yang tersedia
  final cameras = await availableCameras();
  final firstCamera = cameras.first;   // pilih kamera pertama (biasanya depan)

  // jalankan aplikasi
  runApp(MataApp(camera: firstCamera));
}


// =========================================================
//  ROOT APP – membungkus aplikasi dalam MaterialApp
// =========================================================
class MataApp extends StatelessWidget {
  final CameraDescription camera;
  const MataApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Eye Screening PMB',

      // halaman utama aplikasi
      home: MataFormPage(camera: camera),
    );
  }
}


// =========================================================
//  HALAMAN UTAMA – berisi kamera + deteksi + form
// =========================================================
class MataFormPage extends StatefulWidget {
  final CameraDescription camera;
  const MataFormPage({super.key, required this.camera});

  @override
  State<MataFormPage> createState() => _MataFormPageState();
}

class _MataFormPageState extends State<MataFormPage> {
  
  // controller kamera
  late CameraController _controller;

  // proses inisialisasi kamera
  late Future<void> _initializeControllerFuture;

  Timer? _timer;                   // untuk deteksi otomatis
  bool _isDetecting = false;       // supaya tidak double deteksi

  List<double>? pupilCenter;       // posisi pupil kiri
  double? pupilRadius;             // radius pupil (opsional)
  String? hasilDeteksi;            // hasil status dari backend

  // =========================================================
  //  VARIABEL GRAFIK – menampung history pergerakan pupil
  // =========================================================
  List<double> movementLeftHistory = [];   // movement pupil kiri
  List<double> movementRightHistory = [];  // movement pupil kanan

  int maxPoints = 20;                      // tampilkan 20 titik terakhir saja


  // =========================================================
  //  FORM FIELD – data pendaftar
  // =========================================================
  final _formKey = GlobalKey<FormState>();

  final namaCtrl = TextEditingController();
  final emailCtrl = TextEditingController();
  final alamatCtrl = TextEditingController();
  final pekerjaanCtrl = TextEditingController();
  final waCtrl = TextEditingController();
  final sekolahCtrl = TextEditingController();
  final tglLahirCtrl = TextEditingController();

  String jenisKelamin = "Perempuan";
  String prodiDipilih = "Teknik Informatika";

  final List<String> prodiList = [
    "Teknik Informatika",
    "Sistem Informasi",
    "Manajemen",
    "Akuntansi",
    "Teknik Elektro",
    "Hukum",
  ];

  // ID unik untuk client
  String clientId = "client_${DateTime.now().millisecondsSinceEpoch}";


  // =========================================================
  //  ALAMAT BACKEND – server Flask kamu
  // =========================================================
  String _getBaseUrl() {
    return "http://192.168.54.189:5000/"; // ganti sesuai IP backend Yang sedang dijalankan soalnya bisa berubah setiap ganti ip jaringan
  }

  late String baseUrl;


  // =========================================================
  //  INITSTATE – berjalan pertama kali saat halaman dibuka
  // =========================================================
  @override
  void initState() {
    super.initState();

    baseUrl = _getBaseUrl();

    // aktifkan kamera
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,   // kualitas sedang
      enableAudio: false,
    );

    // setelah kamera siap → jalankan deteksi otomatis
    _initializeControllerFuture = _controller.initialize();
    _initializeControllerFuture.then((_) => _startAutoDetect());
  }


  // matikan timer & kamera saat halaman ditutup
  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    super.dispose();
  }


  // =========================================================
  //  DETEKSI OTOMATIS SETIAP 3 DETIK
  // =========================================================
  void _startAutoDetect() {
    _timer = Timer.periodic(Duration(seconds: 3), (_) {
      if (!_isDetecting) _deteksiMata();
    });
  }


  // =========================================================
  //  AMBIL FOTO, KIRIM KE BACKEND, TERIMA HASIL MEDIAPIPE
  // =========================================================
  Future<void> _deteksiMata() async {
    if (_isDetecting) return;

    setState(() => _isDetecting = true);

    try {
      // ambil foto dari kamera
      final XFile picture = await _controller.takePicture();
      final bytes = await picture.readAsBytes();

      final uri = Uri.parse("$baseUrl/detect");
      dynamic responseData;

      // -------------------------------------
      // Mode web → kirim base64
      // -------------------------------------
      if (kIsWeb) {
        final payload = jsonEncode({
          "image": "data:image/jpeg;base64,${base64Encode(bytes)}",
          "client_id": clientId,
        });

        final response = await http.post(
          uri, headers: {"Content-Type": "application/json"}, body: payload);

        responseData = jsonDecode(response.body);
      }

      // -------------------------------------
      // Mode Android/iOS → kirim multipart
      // -------------------------------------
      else {
        final mime = lookupMimeType(picture.path) ?? "image/jpeg";

        final request = http.MultipartRequest("POST", uri);
        request.files.add(http.MultipartFile.fromBytes(
          'image', bytes,
          filename: path.basename(picture.path),
          contentType: MediaType.parse(mime),
        ));

        request.fields["client_id"] = clientId;

        final streamed = await request.send();
        final respStr = await streamed.stream.bytesToString();

        responseData = jsonDecode(respStr);
      }

      // jika tidak ada pupil
      if (responseData["status"] == "pupil_not_found") {
        hasilDeteksi = "Pupil tidak ditemukan";
        pupilCenter = null;
        setState(() {});
      } else {
        // jika berhasil → proses datanya
        _prosesHasil(responseData);
      }

    } catch (e) {
      hasilDeteksi = "Error: $e";
    }

    finally {
      setState(() => _isDetecting = false);
    }
  }



  // =========================================================
  //  MENERIMA DATA DARI BACKEND DAN MENYIMPAN KE STATE
  // =========================================================
  void _prosesHasil(dynamic data) {
    
    if (data == null || data["pupil"] == null) {
      hasilDeteksi = "Tidak ditemukan pupil";
      pupilCenter = null;
      setState(() {});
      return;
    }

    final left = data["pupil"]["left"];
    final right = data["pupil"]["right"];

    // posisi pupil pada layar
    pupilCenter = [
      (left["center_x"] as num).toDouble(),
      (left["center_y"] as num).toDouble(),
    ];

    // radius pupil (opsional)
    pupilRadius = (left["radius"] ?? 0).toDouble();

    // movement kiri & kanan
    final movementLeft = (left["movement"] as num).toDouble();
    final movementRight = (right["movement"] as num).toDouble();

    // simpan movement ke grafik
    movementLeftHistory.add(movementLeft);
    movementRightHistory.add(movementRight);

    // agar grafik tidak terlalu panjang
    if (movementLeftHistory.length > maxPoints) movementLeftHistory.removeAt(0);
    if (movementRightHistory.length > maxPoints) movementRightHistory.removeAt(0);

    // status final dari backend (normal / kemungkinan_tunanetra)
    final status = data["status"];

    // teks yang muncul pada layar
    hasilDeteksi =
        "$status\nMovement L: ${movementLeft.toStringAsFixed(4)} | R: ${movementRight.toStringAsFixed(4)}";

    setState(() {});
  }



  // =========================================================
  //  MEMBANGKITKAN GRAFIK PERGERAKAN PUPIL
  // =========================================================
  Widget _buildMovementChart() {
    return Container(
      height: 250,
      padding: EdgeInsets.all(12),

      decoration: BoxDecoration(
        color: Colors.white, 
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: Offset(0, 2),
          ),
        ],
      ),

      // grafik LineChart
      child: LineChart(
        LineChartData(
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(showTitles: true, reservedSize: 30),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
          ),

          // Dua garis: biru = kiri, merah = kanan
          lineBarsData: [
            
            // GARIS PUPIL KIRI
            LineChartBarData(
              spots: List.generate(
                movementLeftHistory.length,
                (i) => FlSpot(i.toDouble(), movementLeftHistory[i]),
              ),
              isCurved: true,
              color: Colors.blue,
              barWidth: 3,
              dotData: FlDotData(show: false),
            ),

            // GARIS PUPIL KANAN
            LineChartBarData(
              spots: List.generate(
                movementRightHistory.length,
                (i) => FlSpot(i.toDouble(), movementRightHistory[i]),
              ),
              isCurved: true,
              color: Colors.red,
              barWidth: 3,
              dotData: FlDotData(show: false),
            ),
          ],
        ),
      ),
    );
  }



  // =========================================================
  //  UI UTAMA – KAMERA, HASIL DETEKSI, GRAFIK, FORM
  // =========================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFE7F3FA),

      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(20),

          child: Column(
            children: [

              // judul
              Text(
                "Screening Mata & Form PMB",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),

              SizedBox(height: 16),

              // kamera
              FutureBuilder(
                future: _initializeControllerFuture,

                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    
                    return Stack(
                      children: [

                        // tampilan kamera
                        CameraPreview(_controller),

                        // titik merah di pupil
                        if (pupilCenter != null)
                          Positioned(
                            left: pupilCenter![0] * 0.5,
                            top: pupilCenter![1] * 0.5,

                            child: Container(
                              width: 12,
                              height: 12,
                              decoration: BoxDecoration(
                                color: Colors.red,
                                shape: BoxShape.circle,
                              ),
                            ),
                          ),
                      ],
                    );
                  }

                  return CircularProgressIndicator();
                },
              ),

              SizedBox(height: 16),

              // hasil deteksi (normal / kemungkinan tunanetra)
              Text(
                hasilDeteksi ?? "Mendeteksi...",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              ),

              SizedBox(height: 20),

              // grafik movement
              if (movementLeftHistory.isNotEmpty)
                _buildMovementChart(),

              SizedBox(height: 20),

              // form pendaftar
              _buildForm(),
            ],
          ),
        ),
      ),
    );
  }



  // =========================================================
  //  FORM PMB
  // =========================================================
  Widget _buildForm() {
    return Form(
      key: _formKey,

      child: Column(
        children: [

          _field("Nama Lengkap", namaCtrl),
          _field("Email", emailCtrl),
          _field("Alamat Lengkap", alamatCtrl),
          _field("Pekerjaan", pekerjaanCtrl),
          _field("No WA", waCtrl),
          _field("Asal Sekolah", sekolahCtrl),

          SizedBox(height: 16),

          // tombol kirim ke halaman hasil
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
                      tglLahir: tglLahirCtrl.text,
                      jenisKelamin: jenisKelamin,
                      prodi: prodiDipilih,
                      hasilDeteksi: hasilDeteksi ?? "-",
                    ),
                  ),
                );
              }
            },
            child: Text("Kirim"),
          ),
        ],
      ),
    );
  }



  // input field form
  Widget _field(String label, TextEditingController c) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(fontWeight: FontWeight.bold)),

        TextFormField(
          controller: c,
          validator: (v) => v == null || v.isEmpty ? "Harus diisi" : null,
        ),

        SizedBox(height: 10),
      ],
    );
  }

}
