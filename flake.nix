{
  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      ref = "nixos-unstable";
    };

    flake-utils = {
      type = "github";
      owner = "numtide";
      repo = "flake-utils";
    };
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };
      in {
        devShell = pkgs.mkShell rec {
          nativeBuildInputs = with pkgs; [
            cmake
            gnumake
          ];

          buildInputs = with pkgs; [
            zlib
            libzip
            opencl-headers
            ocl-icd
          ];

          LD_LIBRARY_PATH = inputs.nixpkgs.lib.makeLibraryPath buildInputs;
        };
      });
}
