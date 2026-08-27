#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
COMMAND=${1:-install}
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neat-evotrader-matplotlib"

say() { printf '%s\n' "[NEAT-EvoTrader] $*"; }
die() { printf '%s\n' "[NEAT-EvoTrader] ERROR: $*" >&2; exit 1; }
need_python() { command -v python3 >/dev/null 2>&1 || die "python3 is required"; python3 -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)' || die "Python 3.11 through 3.13 is required"; }
make_venv() { need_python; if [ ! -x "$PY" ]; then say "creating isolated environment"; python3 -m venv "$VENV" || die "python3-venv is missing; install it with your OS package manager"; fi; }
install_deps() { make_venv; say "installing/repairing pinned dependencies"; "$PY" -m pip install --disable-pip-version-check --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'; "$PIP" install --disable-pip-version-check -r "$ROOT/requirements.txt"; }
verify() { "$PY" -c 'import numpy, matplotlib, PySide6; import numba; print("dependencies: OK")'; (cd "$ROOT" && "$PY" -m compileall -q main.py LIB tests && "$PY" -m unittest discover -s tests -v); }
demo() { mkdir -p "$MPLCONFIGDIR"; (demo_dir=$(mktemp -d "${TMPDIR:-/tmp}/neat-evotrader-demo.XXXXXX"); trap 'rm -rf "$demo_dir"' EXIT; cd "$ROOT" && "$PY" main.py --generations 2 --population 8 --seed 7 --run-dir "$demo_dir/RUN" --best-dir "$demo_dir/BEST"); }
doctor() { need_python; [ -x "$PY" ] || die "environment missing; run ./install.sh install"; verify; say "doctor: PASS"; }
backup() { out=${2:-"$ROOT/neat-evotrader-backup-$(date +%Y%m%dT%H%M%S).tar.gz"}; tar -czf "$out" -C "$ROOT" CONFIG DATA RUNS BEST; say "backup=$out"; }
restore() { archive=${2:-}; [ -n "$archive" ] && [ -f "$archive" ] || die "usage: ./install.sh restore BACKUP.tar.gz"; tar -tzf "$archive" | awk '/(^|\/)\.\.($|\/)|^\// {bad=1} END {exit bad}' || die "unsafe backup paths"; tar -xzf "$archive" -C "$ROOT"; say "restore complete"; }

case "$COMMAND" in
  install|repair) install_deps; verify; demo ;;
  start|restart) make_venv; verify; demo ;;
  stop) say "no background service is used; nothing to stop" ;;
  status) if [ -x "$PY" ]; then "$PY" --version; say "environment: ready"; else say "environment: not installed"; exit 1; fi ;;
  migrate) install_deps; say "no data migration is required" ;;
  backup) backup "$@" ;;
  restore) restore "$@" ;;
  logs) find "$ROOT/RUNS" -maxdepth 2 -type f -print 2>/dev/null || true ;;
  doctor) doctor ;;
  uninstall) rm -rf "$VENV"; say "environment removed; CONFIG, DATA, RUNS, and BEST preserved" ;;
  test) make_venv; verify ;;
  *) die "usage: ./install.sh {install|start|stop|restart|status|repair|migrate|backup [FILE]|restore FILE|logs|doctor|uninstall|test}" ;;
esac
