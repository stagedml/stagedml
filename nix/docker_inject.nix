# This expression defines author's personal development tools and depends on
# some custom Nix packages. It is not important for the project and may be
# safely removed.

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
  ]);
}

