import 'package:flutter/material.dart';

// ======================================================================
//  HALAMAN TAMPILAN HASIL SCREENING MATA
//  Halaman ini muncul setelah pengguna menekan tombol "Kirim" pada form.
//  Menampilkan:
//    - Status hasil deteksi mata (normal / kemungkinan_tunanetra)
//    - Nilai movement pupil kiri & kanan
//    - Data lengkap pendaftar PMB
// ======================================================================

class HasilPage extends StatelessWidget {
  // -------------------------------
  // Data yang dikirim dari halaman sebelumnya
  // -------------------------------
  final String nama;
  final String email;
  final String alamat;
  final String pekerjaan;
  final String wa;
  final String sekolah;
  final String tglLahir;
  final String jenisKelamin;
  final String prodi;

  // hasil deteksi dari backend (main.dart)
  // contoh:
  // normal
  // Movement L: 0.0031 | R: 0.0045
  final String hasilDeteksi;

  const HasilPage({
    super.key,
    required this.nama,
    required this.email,
    required this.alamat,
    required this.pekerjaan,
    required this.wa,
    required this.sekolah,
    required this.tglLahir,
    required this.jenisKelamin,
    required this.prodi,
    required this.hasilDeteksi,
  });

  @override
  Widget build(BuildContext context) {
    // parse hasil deteksi menjadi map terstruktur
    final parsed = _parseHasil(hasilDeteksi);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Hasil Pemeriksaan & Pendaftaran'),
        backgroundColor: const Color(0xFF68A7D9),
      ),

      // background gradien agar tampilan lebih profesional
      body: Container(
        padding: const EdgeInsets.all(18),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFE7F3FA), Color(0xFFCFE9F3)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),

        child: ListView(
          children: [
            const SizedBox(height: 10),

            // -------------------------------
            // KARTU HASIL DETEKSI PUPIL
            // -------------------------------
            _hasilCard(parsed),

            const SizedBox(height: 25),

            // -------------------------------
            // JUDUL SEKSION DATA FORMULIR
            // -------------------------------
            const Text(
              "Data Formulir Pendaftar",
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Color(0xFF023E8A),
              ),
            ),

            const SizedBox(height: 10),

            // -------------------------------
            // DATA PENGGUNA
            // -------------------------------
            _buildRow("Nama Lengkap", nama),
            _buildRow("Email", email),
            _buildRow("Alamat Lengkap", alamat),
            _buildRow("Pekerjaan", pekerjaan),
            _buildRow("Nomor WA / Telepon", wa),
            _buildRow("Asal Sekolah", sekolah),
            _buildRow("Tanggal Lahir", tglLahir),
            _buildRow("Jenis Kelamin", jenisKelamin),
            _buildRow("Program Studi Pilihan", prodi),

            const SizedBox(height: 30),

            // -------------------------------
            // TOMBOL KEMBALI KE FORM
            // -------------------------------
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.arrow_back),
              label: const Text("Kembali ke Form"),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF68A7D9),
                padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ======================================================================
  // 🧩 PARSER OUTPUT — Mengubah teks hasil deteksi menjadi data terstruktur
  // ======================================================================
  Map<String, dynamic> _parseHasil(String rawText) {
    //
    // rawText contohnya:
    //   normal
    //   Movement L: 0.0031 | R: 0.0045
    //
    // baris pertama = status
    // baris kedua   = movement
    //

    final lines = rawText.split("\n");

    // -------------------------------
    // Ambil status (baris pertama)
    // -------------------------------
    String status = lines.isNotEmpty ? lines[0].trim() : "Tidak diketahui";

    // nilai default movement jika tidak ada
    String left = "-";
    String right = "-";

    // -------------------------------
    // Ambil nilai movement pada baris kedua
    // -------------------------------
    if (lines.length > 1) {
      final movementLine = lines[1];

      // regex untuk mencari angka setelah "L:"
      final matchL = RegExp(r'L:\s*([\d\.]+)').firstMatch(movementLine);

      // regex untuk angka setelah "R:"
      final matchR = RegExp(r'R:\s*([\d\.]+)').firstMatch(movementLine);

      left = matchL != null ? matchL.group(1)! : "-";
      right = matchR != null ? matchR.group(1)! : "-";
    }

    // kembalikan data terstruktur
    return {
      "status": status,
      "left": left,
      "right": right,
    };
  }

  // ======================================================================
  // 🎯 KARTU HASIL DETEKSI (Bagian atas halaman)
  // ======================================================================
  Widget _hasilCard(Map<String, dynamic> data) {
    final status = data["status"];
    final left = data["left"];
    final right = data["right"];

    // warna otomatis:
    //   hijau = normal
    //   merah = kemungkinan masalah
    Color cardColor =
        status == "normal" ? Colors.green.shade600 : Colors.red.shade600;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.15),
            blurRadius: 6,
          ),
        ],
      ),

      child: Column(
        children: [
          const Text(
            "HASIL DETEKSI MATA",
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 15),

          // tampilkan status besar
          Text(
            status.toUpperCase(),
            style: const TextStyle(
              color: Colors.white,
              fontSize: 26,
              fontWeight: FontWeight.w700,
            ),
          ),

          const SizedBox(height: 15),

          // tampilkan movement kiri
          Text(
            "Movement Kiri : $left",
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),

          // tampilkan movement kanan
          Text(
            "Movement Kanan : $right",
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
    );
  }

  // ======================================================================
  // ROW FORM — komponen yang menampilkan setiap field data pendaftar
  // ======================================================================
  Widget _buildRow(String label, String value) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(14),

      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),

        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 6,
          ),
        ],
      ),

      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // label kolom kiri
          Expanded(
            flex: 3,
            child: Text(
              label,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                color: Color(0xFF023E8A),
              ),
            ),
          ),

          // nilai field kolom kanan
          Expanded(
            flex: 4,
            child: Text(
              value.isEmpty ? "-" : value,
              style: const TextStyle(
                color: Colors.black87,
                fontSize: 15,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
