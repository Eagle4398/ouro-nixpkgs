{ pkgs, model ? "2.6B", total_ut_steps ? 4 }:
let
  modelSpecs = {
    "2.6B" = {
      commit = "1ed04250da1a9936042725d302e81c8fa2ab5abd";
      repo = "ByteDance/Ouro-2.6B";
      hash = "sha256-aD9yiAGuZBWU8bzuz7nWZVshRblDXEqwfcgh/bwcNmo=";
    };
    "1.4B" = {
      commit = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1";
      repo = "ByteDance/Ouro-1.4B";
      hash = "";
    };
  };

  spec = modelSpecs.${model};

  ouro-model-raw = pkgs.fetchgit {
    url = "https://huggingface.co/${spec.repo}";
    rev = spec.commit;
    fetchLFS = true;
    hash = spec.hash;
  };

  ouro-model = pkgs.runCommand "ouro-model-configured" {
    nativeBuildInputs = [ pkgs.jq ];
  } ''
    mkdir -p $out

    for f in ${ouro-model-raw}/*; do
      ln -s "$f" $out/$(basename "$f")
    done

    rm $out/config.json 

    jq '.total_ut_steps = ${toString total_ut_steps}' \
      ${ouro-model-raw}/config.json > $out/config.json

    rm  $out/modeling_ouro.py
    cp ${./modeling_ouro.py} $out/modeling_ouro.py
  '';
in ouro-model
