import sys
import math
import wave
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import pyaudio  # type: ignore
except ImportError as e:
    pyaudio = None  # handle later

# ----------------------- Utility Functions ----------------------- #

SCALES = {
    "Pentatónica": [0, 2, 4, 7, 9],
    "Mayor":       [0, 2, 4, 5, 7, 9, 11],
    "Menor":       [0, 2, 3, 5, 7, 8, 10],
}
CURRENT_SCALE = SCALES["Pentatónica"]

def _nearest_in_scale(midi_note: int) -> int:
    base = midi_note % 12
    offset_options = sorted(CURRENT_SCALE, key=lambda x: abs(x - base))
    chosen_offset = offset_options[0]
    return midi_note - base + chosen_offset

# Backward compatibility
_nearest_pentatonic = _nearest_in_scale


def equation_to_frequencies(equation: str) -> List[float]:
    """Convert chars to midi, quantize to CURRENT_SCALE, remove dups, add cadences."""
    notes: List[int] = []
    last_midi = None
    for ch in equation:
        midi = (ord(ch) % 48) + 36
        midi = _nearest_in_scale(midi)
        # avoid immediate duplicate
        if midi == last_midi:
            midi += 12 if midi < 72 else -12
        notes.append(midi)
        last_midi = midi

    # impose cadence every 8 notes
    for i in range(7, len(notes), 8):
        tonic = _nearest_in_scale(60)  # C
        dominant = tonic + 7
        notes[i] = dominant if (i//8)%2 else tonic

    freqs = [440.0 * (2 ** ((m - 69) / 12)) for m in notes]
    return freqs


def _square_wave(f: float, t: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(2 * np.pi * f * t))


def _kick(t: np.ndarray) -> np.ndarray:
    env = np.exp(-40 * t)
    return env * np.sin(2 * np.pi * 60 * t)


def _snare(t: np.ndarray) -> np.ndarray:
    env = np.exp(-20 * t)
    noise = np.random.uniform(-1, 1, t.shape)
    return env * noise


def _chord_for_freq(f: float, t: np.ndarray) -> np.ndarray:
    # Square-wave major triad for chiptune feel
    root = _square_wave(f, t)
    third = _square_wave(f * 1.25, t)
    fifth = _square_wave(f * 1.5, t)
    return (root + third + fifth) / 3


def _adsr(t: np.ndarray, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.ones_like(t)
    a_len = int(len(t)*attack)
    d_len = int(len(t)*decay)
    r_len = int(len(t)*release)
    s_len = len(t)-a_len-d_len-r_len
    if a_len>0:
        env[:a_len]=np.linspace(0,1,a_len)
    if d_len>0:
        env[a_len:a_len+d_len]=np.linspace(1,sustain,d_len)
    if s_len>0:
        env[a_len+d_len:a_len+d_len+s_len]=sustain
    if r_len>0:
        env[-r_len:]=np.linspace(sustain,0,r_len)
    return env


def compose_audio(frequencies: List[float], bpm: int = 120, sample_rate: int = 44100) -> np.ndarray:
    """Compose a chiptune-style track with melody, bass, kick, snare."""
    if not frequencies:
        return np.array([], dtype=np.int16)

    base_step = 60 / bpm / 2  # eighth
    segments: List[np.ndarray] = []

    for i, f in enumerate(frequencies):
        # swing factor
        step_duration = base_step * (0.9 if i % 2 == 0 else 1.1)
        t_step = np.arange(int(sample_rate * step_duration)) / sample_rate
        chord = _chord_for_freq(f, t_step) * _adsr(t_step,0.05,0.1,0.7,0.15)
        # bass every downbeat
        if i % 4 == 0:
            chord += 0.6 * _square_wave(f / 2, t_step) * _adsr(t_step,0.05,0.2,0.5,0.2)
        # kick on 1 & 3 (0,8,...)
        if i % 8 == 0:
            chord += 0.8 * _kick(t_step)
        # snare on 2 & 4 (4,12,...)
        if i % 8 == 4:
            chord += 0.4 * _snare(t_step)
        # hi-hat every even step
        if i % 2 == 0:
            hat = _snare(t_step)*0.2
            chord += hat
        segments.append(chord)

    audio = np.concatenate(segments)
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    return audio_int16
    """Generate concatenated harmonic chords for given root frequencies."""
    if not frequencies:
        return np.array([], dtype=np.int16)
    segments = []
    t = np.arange(int(sample_rate * duration)) / sample_rate
    for f in frequencies:
        chord = _chord_for_freq(f, t)
        segments.append(chord)
    audio = np.concatenate(segments)
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    return audio_int16


def image_to_frequencies(img_path: Path) -> List[float]:
    """Convert an image to a list of frequencies by mapping pixel brightness to pitch."""
    try:
        img = Image.open(img_path).convert('L')  # grayscale
    except Exception:
        return []
    # Resize for lighter processing
    img = img.resize((128, 64))
    pixels = np.array(img).flatten()
    freqs: List[float] = []
    for val in pixels[::50]:  # sample every 50th pixel for shorter sequences
        midi = int(val / 255 * 48) + 36
        midi = _nearest_in_scale(midi)
        freq = 440.0 * (2 ** ((midi - 69) / 12))
        freqs.append(freq)
    return freqs


def save_wav(path: Path, audio_data: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


# ----------------------- Worker Thread ----------------------- #

class PlayThread(QThread):
    finished = Signal()

    def __init__(self, audio_data: np.ndarray, sample_rate: int = 44100):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def run(self):
        if pyaudio is None:
            self.finished.emit()
            return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True)
        stream.write(self.audio_data.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.finished.emit()


# ----------------------- Image Drop Widget ----------------------- #

class ImageDropLabel(QLabel):
    """Etiqueta que acepta arrastrar imágenes y emite la ruta."""
    image_dropped = Signal(Path)

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # acepta solo archivos con extensión de imagen
            valid = any(Path(u.toLocalFile()).suffix.lower() in {'.png', '.jpg', '.jpeg'} for u in event.mimeData().urls())
            if valid:
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                self.image_dropped.emit(path)
                break


# ----------------------- Main Window ----------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math ⬌ Music Composer")
        self.setMinimumSize(900, 600)
        self._setup_ui()
        self.audio_data: np.ndarray = None
        self.image_path: Path = None
        self.play_thread: PlayThread = None

    def _setup_ui(self):
        # Central splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel – input
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.equation_edit = QTextEdit()
        self.equation_edit.setPlaceholderText("Escribe una ecuación o expresión matemática…")
        left_layout.addWidget(QLabel("Ecuación"))
        left_layout.addWidget(self.equation_edit)
        # Scale selector
        self.scale_box = QComboBox()
        for name in SCALES.keys():
            self.scale_box.addItem(name)
        self.scale_box.currentTextChanged.connect(self.change_scale)
        left_layout.addWidget(QLabel("Escala"))
        left_layout.addWidget(self.scale_box)

        self.image_label = ImageDropLabel("Arrastra una imagen PNG/JPG aquí")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #666; padding: 20px;")
        self.image_label.setAcceptDrops(True)
        left_layout.addWidget(self.image_label)
        self.image_label.image_dropped.connect(self.load_image)
        browse_btn = QPushButton("📂 Abrir imagen…")
        browse_btn.clicked.connect(self.browse_image)
        left_layout.addWidget(browse_btn)

        self.generate_btn = QPushButton("Generar Composición")
        self.generate_btn.clicked.connect(self.generate_music)
        left_layout.addWidget(self.generate_btn)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # Right panel – explanation + controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.explanation_view = QTextEdit()
        self.explanation_view.setReadOnly(True)
        right_layout.addWidget(QLabel("Explicación"))
        right_layout.addWidget(self.explanation_view)

        self.play_btn = QPushButton("▶️ Reproducir")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        right_layout.addWidget(self.play_btn)

        self.export_btn = QPushButton("💾 Exportar WAV")
        self.export_btn.clicked.connect(self.export_audio)
        self.export_btn.setEnabled(False)
        right_layout.addWidget(self.export_btn)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        self.setCentralWidget(splitter)

        # Dark style
        self.setStyleSheet("""
            QWidget { background: #222; color: #ddd; font-family: Segoe UI, sans-serif; }
            QLineEdit, QTextEdit { background: #333; color: #eee; border: 1px solid #555; }
            QPushButton { background: #444; border: 1px solid #666; padding: 6px 12px; }
            QPushButton:disabled { background: #555; color: #888; }
            QLabel { font-size: 13px; }
        """)

        # Menu (optional future use)
        export_action = QAction("Exportar WAV", self)
        export_action.triggered.connect(self.export_audio)
        file_menu = self.menuBar().addMenu("Archivo")
        file_menu.addAction(export_action)

    # ----------------------- Slots ----------------------- #



    def play_audio(self):
        if self.audio_data is None:
            return
        self.play_btn.setEnabled(False)
        thread = PlayThread(self.audio_data)
        def _on_finished():
            self.play_btn.setEnabled(True)
            self.play_thread = None
        thread.finished.connect(_on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        # Mantener referencia viva hasta que termine
        self.play_thread = thread

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path):
        self.image_path = path
        pixmap = QPixmap(str(path)).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

    def change_scale(self, name: str):
        global CURRENT_SCALE
        CURRENT_SCALE = SCALES[name]

    def generate_music(self):
        equation_text = self.equation_edit.toPlainText().strip()
        if equation_text:
            freqs = equation_to_frequencies(equation_text)
            explanation = (f"Entrada: ecuación con {len(equation_text)} caracteres\n"
                           f"Notas generadas: {', '.join(f'{round(f,1)} Hz' for f in freqs[:10])}{'…' if len(freqs)>10 else ''}")
        elif self.image_path:
            freqs = image_to_frequencies(self.image_path)
            explanation = (f"Entrada: imagen {self.image_path.name}\nTamaño mapeado a {len(freqs)} notas")
        else:
            QMessageBox.warning(self, "Vacío", "Por favor escribe una ecuación o carga una imagen.")
            return

        if not freqs:
            QMessageBox.critical(self, "Error", "No se pudieron generar notas a partir de la entrada.")
            return

        self.audio_data = compose_audio(freqs)
        self.explanation_view.setPlainText(explanation)
        self.play_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def export_audio(self):
        if self.audio_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar WAV", "composition.wav", "WAV (*.wav)")
        if path:
            save_wav(Path(path), self.audio_data)
            QMessageBox.information(self, "Guardado", f"Archivo guardado en {path}")


# ----------------------- Entry Point ----------------------- #

    def closeEvent(self, event):
        if self.play_thread is not None and self.play_thread.isRunning():
            self.play_thread.wait()
        super().closeEvent(event)


# ----------------------- Entry Point ----------------------- #

def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
