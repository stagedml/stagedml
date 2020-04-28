# This expression defines author's personal development tools and depends on
# custom Nix packages. It is not an important part of the project and may be
# safely ignored.

{ me
, localpkgs ?  import <nixcfg/src/pkgs/localpkgs.nix> {inherit me;}
, pkgs ? localpkgs.nixpkgs
, stdenv ? pkgs.stdenv
} :
pkgs.buildEnv {
  name = "docker-inject";
  paths = []
  ++(with localpkgs; [
    myvim
    mytmux
    myprofile
  ])
  ++(with pkgs; [
    gitFull
    ccls
    fzf
    fd
  ]);
}

