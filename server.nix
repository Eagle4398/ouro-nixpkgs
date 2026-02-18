{
  pkgs,
  quantization_bits ? null,
  model ? "2.6B",
  ...
}:
let
  ouro-model = import ./ouro-model.nix {
    inherit pkgs;
    model = model;
  };

  quantConfig =
    if quantization_bits == 8 then
      ''
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
      ''
    else if quantization_bits == 4 then
      ''
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
      ''
    else
      ''
        quantization_config = None
      '';

  modelLoadArgs =
    if quantization_bits != null then
      ''
        quantization_config=quantization_config,
      ''
    else
      "";

  transformers-pin = pkgs.python3.override {
    packageOverrides = self: super: {
      bitsandbytes = super.bitsandbytes.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.ninja ];
      });
      transformers = super.transformers.overridePythonAttrs (old: {
        version = "4.54.1";
        src = super.fetchPypi {
          pname = "transformers";
          version = "4.54.1";
          hash = "sha256-slUbuXkD8TvZDJRn0KFE1Byk0ULe/ARKmVArt3xcEFI=";
        };
        doCheck = false;
      });
      tokenizers = super.tokenizers.overridePythonAttrs (old: rec {
        version = "0.21.0";
        src = pkgs.fetchFromGitHub {
          owner = "huggingface";
          repo = "tokenizers";
          tag = "v${version}";
          hash = "sha256-G65XiVlvJXOC9zqcVr9vWamUnpC0aa4kyYkE2v1K2iY=";
        };
        cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
          pname = "tokenizers";
          inherit version src;
          sourceRoot = "${src.name}/bindings/python";
          hash = "sha256-jj5nuwxlfJm1ugYd5zW+wjyczOZHWCmRGYpmiMDqFlk=";
        };
        doCheck = false;
      });
      torch = super.torch-bin.overridePythonAttrs (old: {
        passthru = (old.passthru or { }) // {
          cudaSupport = true;
          cudaPackages = pkgs.cudaPackages;
          rocmSupport = false;
          rocmPackages = { };
        };
      });
    };
  };

  ouro-server-script =
    pkgs.writers.makePythonWriter transformers-pin # set interpreter
      transformers-pin.pkgs # package set
      transformers-pin.pkgs # build-time package set
      "ouro-server"
      {
        libraries =
          with transformers-pin.pkgs;
          [
            transformers
            accelerate
            torch-bin
            flask
          ]
          ++ pkgs.lib.optionals (quantization_bits != null) [ bitsandbytes ];
        flakeIgnore = [
          "E501"
          "E302"
          "E305"
          "E303"
          "E226"
          "W293"
          "W291"
        ];
      }
      ''
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ${pkgs.lib.optionalString (quantization_bits != null) "from transformers import BitsAndBytesConfig"}
        from flask import Flask, request, jsonify
        import torch

        app = Flask(__name__)

        print(f"CUDA available: {torch.cuda.is_available()}")

        ${quantConfig}

        model = AutoModelForCausalLM.from_pretrained(
            "${ouro-model}",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            ${modelLoadArgs}
        )
        tokenizer = AutoTokenizer.from_pretrained("${ouro-model}")


        print(f"[INIT] Model dtype: {model.dtype}")
        print(f"[INIT] First layer dtype: {next(model.parameters()).dtype}")

        print(f"[INIT] Model config: {model.config}")
        print(f"[INIT] Model architecture: {model.config.architectures}")

        print(f"[INIT] Tokenizer chat template: {tokenizer.chat_template}")
        print(f"[INIT] EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
        print(f"[INIT] PAD token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")

        print(f"[INIT] <|im_end|> token id: {tokenizer.convert_tokens_to_ids('<|im_end|>')}")


        print(f"[INIT] Token 31: {repr(tokenizer.decode([31]))}")
        print(f"[INIT] Token 33: {repr(tokenizer.decode([33]))}") 
        print(f"[INIT] Token 49: {repr(tokenizer.decode([49]))}")

        @app.route('/generate', methods=['POST'])
        def generate():
            prompt = request.json.get('prompt', "")

            print(f"\n{'='*60}")
            print(f"[REQUEST] Raw prompt: {repr(prompt)}")

            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"[FORMATTED] After chat template: {repr(formatted)}")

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            print(f"[TOKENIZED] Token count: {inputs.shape[-1]}")
            print(f"[TOKENIZED] Token IDs: {inputs[0].tolist()}")

            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )

            new_token_ids = outputs[0][inputs.shape[-1]:]
            print(f"[OUTPUT] Raw new token IDs: {new_token_ids.tolist()}")
            new_decoded = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

            print(f"[OUTPUT] New tokens only ({len(new_token_ids)} tokens): {repr(new_decoded)}")
            print(f"{'='*60}\n")

            return jsonify({"response": new_decoded})

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

in
pkgs.stdenv.mkDerivation {
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
