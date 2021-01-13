{
  "targets": [
    {
      "target_name": "ws_sendframe_js",
      "cflags!": [ "-fno-exceptions", "-fno-rtti" ],
      "cflags_cc!": [ "-fno-exceptions", "-fno-rtti" ],
      "cflags_cc": [ "-Wno-misleading-indentation", "-std=c++17" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.7"
      },
      "msvs_settings": {
        "VCCLCompilerTool": { "ExceptionHandling": 1 },
      },
      "sources": [
        "./src/ws_sendframe_js.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../include",
        "../../external/rnf/include",
        "../../dist/include"
      ],
      'defines': [ 'S3DMM_REAL_TYPE=float', 'NAPI_CPP_EXCEPTIONS' ],
      "libraries": [
            "-L <!@(cd ../../builds/revise/release/bin; pwd)", "-lws_sendframe",
            "-lboost_system", "-lboost_filesystem"
      ]
    }
  ]
}
