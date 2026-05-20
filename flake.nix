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
        pythonPackages = python.pkgs;
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (python.withPackages (
              python-pkgs: with python-pkgs; [
                opencv4
                torch
                torchvision
              ]
            ))
          ];

          shellHook = ''
            echo "Python development environment"
            echo "Python version: $(python --version)"
          '';
        };
      }
    );
}
