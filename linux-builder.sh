#!/usr/bin/env bash
set -xe

mkdir -p "$BUILD_DIR" "$ARTIFACT_DIR"

cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_PREFIX_PATH="$DEPS_DIR" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-fopenmp"

cmake --build . --target install -j$(nproc)

# run tests
if [ -f "$BUILD_DIR/lib/unit" ]; then
    "$BUILD_DIR/lib/unit"
fi

# collect artifacts
mkdir -p "$ARTIFACT_DIR/linux"
cp -v "$BUILD_DIR/src/main" "$ARTIFACT_DIR/linux/" || true
cp -v "$BUILD_DIR/lib/unit" "$ARTIFACT_DIR/linux/" || true
cp -v "$BUILD_DIR/lib/benchmark" "$ARTIFACT_DIR/linux/" || true
cp -rvL "$BUILD_DIR/install/"* "$ARTIFACT_DIR/linux/" || true
# find "$BUILD_DIR" -name "*.so" -exec cp -v {} "$ARTIFACT_DIR/linux/" \; || true

echo "Artifacts:"
ls -lR "$ARTIFACT_DIR/linux"
