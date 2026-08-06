# Makefile for the WebGPU triangle example (wgpu-native + GLFW).
#
# Dependencies:
#   - GLFW3, found via pkg-config.
#       macOS:  brew install glfw
#       Debian: sudo apt install libglfw3-dev pkg-config unzip
#   - wgpu-native: downloaded automatically (prebuilt binary + headers) into
#     deps/wgpu the first time you build.
#
# Usage:
#   make            # download deps (once) and build ./triangle
#   make run        # build and run
#   make clean      # remove build artifacts
#   make distclean  # also remove downloaded dependencies

WGPU_VERSION := v29.0.1.1
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    WGPU_ASSET := wgpu-macos-aarch64-release.zip
  else
    WGPU_ASSET := wgpu-macos-x86_64-release.zip
  endif
else ifeq ($(UNAME_S),Linux)
  ifeq ($(UNAME_M),aarch64)
    WGPU_ASSET := wgpu-linux-aarch64-release.zip
  else
    WGPU_ASSET := wgpu-linux-x86_64-release.zip
  endif
else
  $(error Unsupported platform $(UNAME_S)/$(UNAME_M) -- see https://github.com/gfx-rs/wgpu-native/releases)
endif

DEPS_DIR := deps
WGPU_DIR := $(DEPS_DIR)/wgpu
WGPU_STAMP := $(WGPU_DIR)/.stamp

CC := cc
CXX := c++
GLFW_CFLAGS := $(shell pkg-config --cflags glfw3)
GLFW_LIBS := $(shell pkg-config --libs glfw3)

CFLAGS := -std=c11 -Wall -Wextra -I. -I$(WGPU_DIR)/include $(GLFW_CFLAGS)
CXXFLAGS := -std=c++17 -Wall -Wextra -I. -I$(WGPU_DIR)/include $(GLFW_CFLAGS)
# linked statically (the .a) to avoid runtime dylib/rpath lookup issues
LDFLAGS := $(WGPU_DIR)/lib/libwgpu_native.a $(GLFW_LIBS)

# glfw3webgpu.c needs to know which native backend GLFW itself was built
# with (these macros are normally private to GLFW's own build), and on
# macOS it uses Objective-C to talk to Cocoa/Metal.
ifeq ($(UNAME_S),Darwin)
  GLFW3WEBGPU_FLAGS := -x objective-c -D_GLFW_COCOA
  LDFLAGS += -framework Cocoa -framework CoreVideo -framework IOKit -framework QuartzCore -framework Metal
else
  GLFW3WEBGPU_FLAGS := -D_GLFW_X11 -D_GLFW_WAYLAND
  LDFLAGS += -pthread -ldl -lm
endif

OBJS := triangle.o glfw3webgpu.o

.PHONY: all run clean distclean deps

all: triangle

triangle: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

triangle.o: triangle.cpp $(WGPU_STAMP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

glfw3webgpu.o: glfw3webgpu.c glfw3webgpu.h $(WGPU_STAMP)
	$(CC) $(CFLAGS) $(GLFW3WEBGPU_FLAGS) -c $< -o $@

deps: $(WGPU_STAMP)

# Downloads and unpacks the prebuilt wgpu-native library + headers.
$(WGPU_STAMP):
	mkdir -p $(WGPU_DIR)
	curl -L -o $(DEPS_DIR)/wgpu.zip https://github.com/gfx-rs/wgpu-native/releases/download/$(WGPU_VERSION)/$(WGPU_ASSET)
	unzip -o -q $(DEPS_DIR)/wgpu.zip -d $(WGPU_DIR)
	rm -f $(DEPS_DIR)/wgpu.zip
	touch $@

run: triangle
	./triangle

clean:
	rm -f triangle $(OBJS)

distclean: clean
	rm -rf $(DEPS_DIR)
