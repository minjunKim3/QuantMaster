@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist C:\t mkdir C:\t
set "TMP=C:\t"
set "TEMP=C:\t"
set "CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON -DGGML_F16C=ON"
set FORCE_CMAKE=1
"C:\QuantMaster\server\agent\AItr_manager\venv\Scripts\python.exe" -m pip install llama-cpp-python==0.3.23 --no-binary llama-cpp-python --force-reinstall --no-cache-dir
