{
  "targets": [
    {
      "target_name": "tree_sitter_simplex_binding",
      "include_dirs": [
        "<!(node -e \"require('node-addon-api').include\")",
        "src"
      ],
      "sources": [
        "bindings/node/binding.cc",
        "src/parser.c"
      ],
      "cflags_c": [
        "-std=c11",
        "-fvisibility=hidden"
      ],
      "cflags_cc": [
        "-fvisibility=hidden"
      ],
      "xcode_settings": {
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "MACOSX_DEPLOYMENT_TARGET": "10.15",
        "OTHER_CFLAGS": [
          "-fvisibility=hidden"
        ]
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "AdditionalOptions": [
            "/std:c17"
          ]
        }
      },
      "conditions": [
        [
          "OS=='win'",
          {
            "defines": [
              "_CRT_SECURE_NO_WARNINGS"
            ]
          }
        ]
      ]
    }
  ]
}
