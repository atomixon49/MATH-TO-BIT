# Math ⬌ Music Composer

Una aplicación que convierte ecuaciones matemáticas e imágenes en música usando diferentes escalas musicales.

## Características

- **Conversión de ecuaciones**: Transforma caracteres matemáticos en notas musicales
- **Procesamiento de imágenes**: Convierte imágenes en frecuencias basadas en el brillo de píxeles
- **Múltiples escalas**: Pentatónica, Mayor, Menor, Armónica, Melódica, Dórica, Mixolidia, Blues
- **Composición automática**: Genera pistas completas con melodía, bajo, batería y efectos
- **Reproducción SF2**: Soporte para SoundFonts con calidad de instrumentos reales
- **Exportación WAV**: Guarda las composiciones como archivos de audio

## Instalación

### Requisitos

- Python 3.8+
- PySide6 (interfaz gráfica)
- NumPy (procesamiento de audio)
- PyAudio (reproducción de audio)
- Pillow (procesamiento de imágenes)
- Mido (manipulación MIDI)
- PyFluidSynth (reproducción SF2)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### SoundFonts (opcional)

Para usar la función "Reproducir SF2", coloca un archivo `.sf2` en el directorio del proyecto:
- `guzheng.SF2` (incluido)
- O descarga otros SoundFonts de [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont)

## Uso

### Ejecutar la aplicación

```bash
python main.py
```

### Interfaz

1. **Panel izquierdo - Entrada**:
   - Campo de texto para ecuaciones matemáticas
   - Selector de escala musical
   - Área para arrastrar imágenes
   - Botón para generar composición

2. **Panel derecho - Control**:
   - Explicación de la composición generada
   - Botón de reproducción
   - Exportación a WAV
   - Reproducción con SoundFont

### Ejemplos de uso

#### Ecuaciones matemáticas
```
2x + 3y = 7
∫ f(x) dx
√(a² + b²)
```

#### Imágenes
- Arrastra archivos PNG/JPG al área designada
- La aplicación mapeará el brillo de píxeles a frecuencias

## Actualización FluidSynth 1.3+

### Cambios realizados

La aplicación ha sido actualizada para ser compatible con la nueva API de FluidSynth 1.3+:

- **Eliminación de Player**: La clase `Player` fue removida en la versión 1.3+
- **Nuevo Sequencer**: Implementación usando `Sequencer` para reproducción MIDI
- **Múltiples fallbacks**: El código intenta diferentes métodos de reproducción
- **Manejo robusto de errores**: Mejor gestión de problemas de inicialización

### Métodos de reproducción

1. **Sequencer (nueva API)**: Método principal para FluidSynth 1.3+
2. **Player (API antigua)**: Fallback para versiones anteriores
3. **Reproducción manual**: Usando mido para control directo de notas

### Verificación de compatibilidad

Ejecuta el script de prueba para verificar que todo funciona:

```bash
python test_fluidsynth.py
```

## Estructura del código

```
main.py              # Aplicación principal
requirements.txt     # Dependencias
test_fluidsynth.py   # Script de prueba
guzheng.SF2         # SoundFont de ejemplo
README.md           # Este archivo
```

## Escalas musicales

- **Pentatónica**: [0,2,4,7,9] - Escala básica de 5 notas
- **Mayor**: [0,2,4,5,7,9,11] - Escala mayor tradicional
- **Menor**: [0,2,3,5,7,8,10] - Escala menor natural
- **Menor Armónica**: [0,2,3,5,7,8,11] - Menor con 7ª aumentada
- **Menor Melódica**: [0,2,3,5,7,9,11] - Menor melódica ascendente
- **Dórica**: [0,2,3,5,7,9,10] - Modo dórico
- **Mixolidia**: [0,2,4,5,7,9,10] - Modo mixolidio
- **Blues Mayor**: [0,2,3,4,7,9] - Escala blues mayor
- **Blues Menor**: [0,3,5,6,7,10] - Escala blues menor

## Solución de problemas

### Error de FluidSynth
Si aparece un error relacionado con SDL o inicialización:
- La aplicación intentará múltiples métodos de reproducción
- Verifica que tienes un archivo `.sf2` en el directorio
- Ejecuta `python test_fluidsynth.py` para diagnosticar

### Problemas de audio
- Asegúrate de que PyAudio esté instalado correctamente
- En Windows, puede requerir Microsoft Visual C++ Build Tools

### Dependencias faltantes
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Licencia

Este proyecto es de código abierto. Contribuciones son bienvenidas.

## Créditos

- **FluidSynth**: Motor de síntesis de audio
- **PySide6**: Framework de interfaz gráfica
- **Mido**: Biblioteca para manipulación MIDI 