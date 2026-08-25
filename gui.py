"""PySide6 control panel for deterministic NEAT research."""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def demo_command(run_dir: str) -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), str(ROOT / "DATA/sample_ohlc.csv"), "--generations", "2", "--population", "8", "--seed", "7", "--run-dir", run_dir]


def run_demo() -> dict:
    with tempfile.TemporaryDirectory(prefix="neat-demo-") as tmp:
        proc = subprocess.run(demo_command(tmp), cwd=ROOT, text=True, capture_output=True, check=False)
        return {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "mode": "synthetic paper simulation", "output": proc.stdout, "errors": proc.stderr}


def launch() -> int:
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication, QFormLayout, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget
    class Window(QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle("NEAT-EvoTrader Research Control Panel"); self.resize(940, 650)
            self.process = QProcess(self); root=QWidget(); layout=QVBoxLayout(root)
            layout.addWidget(QLabel("Offline historical/synthetic research only — no live trading connection exists."))
            form=QFormLayout(); self.generations=QSpinBox(); self.generations.setRange(1,10000); self.generations.setValue(20)
            self.population=QSpinBox(); self.population.setRange(4,10000); self.population.setValue(48)
            self.seed=QSpinBox(); self.seed.setRange(0,2147483647); self.seed.setValue(7)
            form.addRow("Generations",self.generations); form.addRow("Population",self.population); form.addRow("Seed",self.seed); layout.addLayout(form)
            row=QHBoxLayout(); self.demo=QPushButton("Run Safe Demo"); self.start=QPushButton("Start Research Run"); self.stop=QPushButton("Stop")
            row.addWidget(self.demo); row.addWidget(self.start); row.addWidget(self.stop); layout.addLayout(row)
            self.progress=QProgressBar(); layout.addWidget(self.progress); self.status=QLabel("Ready"); layout.addWidget(self.status)
            self.output=QPlainTextEdit(); self.output.setReadOnly(True); layout.addWidget(self.output); self.setCentralWidget(root)
            self.demo.clicked.connect(self.run_safe_demo); self.start.clicked.connect(self.run_job); self.stop.clicked.connect(self.process.kill)
            self.process.readyReadStandardOutput.connect(self.read_output); self.process.readyReadStandardError.connect(self.read_error); self.process.finished.connect(self.finished)
        def begin(self,args):
            if self.process.state()!=QProcess.NotRunning:return
            self.output.clear(); self.status.setText("Running"); self.progress.setRange(0,0); self.process.setWorkingDirectory(str(ROOT)); self.process.start(sys.executable,args)
        def run_safe_demo(self):
            target=str(ROOT/"RUNS"/"gui-demo"); self.begin([str(ROOT/"main.py"),str(ROOT/"DATA/sample_ohlc.csv"),"--generations","2","--population","8","--seed","7","--run-dir",target])
        def run_job(self):
            self.begin([str(ROOT/"main.py"),str(ROOT/"DATA/sample_ohlc.csv"),"--generations",str(self.generations.value()),"--population",str(self.population.value()),"--seed",str(self.seed.value())])
        def read_output(self): self.output.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors="replace").rstrip())
        def read_error(self): self.output.appendPlainText(bytes(self.process.readAllStandardError()).decode(errors="replace").rstrip())
        def finished(self,code,_): self.progress.setRange(0,100); self.progress.setValue(100 if code==0 else 0); self.status.setText("Complete" if code==0 else f"Failed ({code})")
    app=QApplication([]); window=Window(); window.show(); return app.exec()


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--demo",action="store_true"); a=p.parse_args()
    if a.demo:
        result=run_demo(); print(json.dumps(result,indent=2)); return 0 if result["status"]=="PASS" else 1
    return launch()
if __name__=="__main__": raise SystemExit(main())
