if [ $(uname) == "Darwin" ]; then
    cmake -B build . \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenMP_C_FLAGS=-fopenmp=lomp \
        -DOpenMP_CXX_FLAGS=-fopenmp=lomp \
        -DOpenMP_C_LIB_NAMES="libomp" \
        -DOpenMP_CXX_LIB_NAMES="libomp" \
        -DOpenMP_libomp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include" \
        -DOpenMP_CXX_LIB_NAMES="libomp" \
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include"

    cmake --build build

    make -C build -j faiss
fi
