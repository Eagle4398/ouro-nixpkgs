{ pkgs, quantization_bits ? null, model ? "2.6B", ... }:
let
  ouro-model = import ./ouro-model.nix {
    inherit pkgs;
    model = model;
  };

  quantConfig = if quantization_bits == 8 then ''
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  '' else if quantization_bits == 4 then ''
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
  '' else ''
    quantization_config = None
  '';

  modelLoadArgs = if quantization_bits != null then ''
    quantization_config=quantization_config,
  '' else
    "";

  # I do not know who I need to murder but why is libnvshmem, which is for 
  # multi-GPU training not cached ANYWHERE. I am skipping this as this package 
  # is geniunely only for local single-gpu inference
  torch-bin-custom = pkgs.python3.pkgs.torch-bin.overridePythonAttrs (old: {
    buildInputs = pkgs.lib.filter (x: x != pkgs.cudaPackages.libnvshmem)
      (old.buildInputs or [ ]); 
    autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ])
      ++ [ "libnvshmem.so.3" ];
  });

  python-env = pkgs.python3.withPackages (ps:
    [
      (ps.transformers.overridePythonAttrs (old: rec {
        version = "4.54.1";
        src = ps.fetchPypi {
          pname = "transformers";
          inherit version;
          hash = "";
        };
      }))
      torch-bin-custom
      ps.flask
    ] ++ pkgs.lib.optional (quantization_bits != null) ps.bitsandbytes);

  ouro-server-script = pkgs.writers.writePython3 "ouro-server" {
    libraries = with python-env.pkgs;
      [ transformers torch-bin-custom flask ]
      ++ pkgs.lib.optional (quantization_bits != null)
      python-env.pkgs.bitsandbytes;
  } ''
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ${pkgs.lib.optionalString (quantization_bits != null)
    "from transformers import BitsAndBytesConfig"}
    from flask import Flask, request, jsonify
    import torch

    app = Flask(__name__)

    print(f"CUDA available: {torch.cuda.is_available()}")

    ${quantConfig}

    model = AutoModelForCausalLM.from_pretrained(
        "${ouro-model}",
        device_map="auto",
        torch_dtype="auto",
        ${modelLoadArgs}
    )
    tokenizer = AutoTokenizer.from_pretrained("${ouro-model}")

    @app.route('/generate', methods=['POST'])
    def generate():
        prompt = request.json.get('prompt', "")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response})

    if __name__ == '__main__':
        app.run(host='localhost', port=8000)
  '';

  ouro-query-script = pkgs.writeShellScript "ouro-query" ''
    PROMPT=$(cat)
    ${pkgs.curl}/bin/curl -s -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d "$(${pkgs.jq}/bin/jq -n --arg prompt "$PROMPT" '{prompt: $prompt}')" \
      | ${pkgs.jq}/bin/jq -r '.response'
  '';

in pkgs.stdenv.mkDerivation {
  pname = "ouro-tools";
  version = "0.1.0";

  nativeBuildInputs = [ pkgs.makeWrapper ];

  dontUnpack = true;

  installPhase = ''
    mkdir -p $out/bin
    cp ${ouro-server-script} $out/bin/ouro-server
    cp ${ouro-query-script} $out/bin/ouro-query
    chmod +x $out/bin/*
  '';
}

