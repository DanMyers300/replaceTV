{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        pythonEnv = python.withPackages (python-pkgs: with python-pkgs; [
          opencv4
          numpy
          pip
        ]);
        script = pkgs.writeShellScriptBin "replaceTV" ''
          export PIP_PREFIX="''${HOME}/.cache/replaceTV-pip"
          export PYTHONPATH="''${PIP_PREFIX}/lib/python3.12/site-packages:''${PYTHONPATH:-}"
          export PATH="''${PIP_PREFIX}/bin:''${PATH}"
          mkdir -p "''${PIP_PREFIX}/lib/python3.12/site-packages"

          if ! ${pythonEnv}/bin/python -c "import fal_client" 2>/dev/null; then
            echo "Installing fal-client..."
            ${pythonEnv}/bin/pip install fal-client --prefix="''${PIP_PREFIX}" --quiet
          fi

          exec ${pythonEnv}/bin/python ${self}/run.py --input_dir ./input_images --output_dir ./output_images "$@"
        '';
      in
      {
        apps.default = {
          type = "app";
          program = "${script}/bin/replaceTV";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];

          shellHook = ''
            echo "Python development environment"
            echo "Python version: $(python --version)"

            # fal-client is not yet packaged in nixpkgs — install to a local prefix
            export PIP_PREFIX="$PWD/.pip-packages"
            export PYTHONPATH="$PIP_PREFIX/lib/python3.12/site-packages:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            mkdir -p "$PIP_PREFIX/lib/python3.12/site-packages"

            if ! python -c "import fal_client" 2>/dev/null; then
              echo "Installing fal-client..."
              pip install fal-client --prefix="$PIP_PREFIX" --quiet
            fi
          '';
        };
      }
    );
}
