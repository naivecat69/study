# -*- coding: utf-8 -*-
from __future__ import print_function
from PyQt4 import QtCore, QtGui
import sys, time, os

# ---------------- Worker (백그라운드 작업 전담) ----------------
class Worker(QtCore.QObject):
    progressed = QtCore.pyqtSignal(int)   # 0~100
    message    = QtCore.pyqtSignal(str)   # 로그 텍스트
    finished   = QtCore.pyqtSignal()      # 완료 시그널

    def __init__(self, dir_path):
        super(Worker, self).__init__()
        self.dir_path = dir_path

    @QtCore.pyqtSlot()
    def run(self):
        """긴 작업: 여기서 '메인 함수' 역할을 한다고 보면 됨."""
        try:
            self.message.emit("작업 시작: {}".format(self.dir_path or "(폴더 미선택)"))
            # 예시: 선택한 폴더 안 파일 개수를 세면서 진행률 업데이트
            files = []
            if self.dir_path and os.path.isdir(self.dir_path):
                for root, _, fnames in os.walk(self.dir_path):
                    for f in fnames:
                        files.append(os.path.join(root, f))
            total = max(1, len(files))  # 0으로 나눔 방지

            for i, path in enumerate(files or range(100)):  # 파일이 없으면 더미 루프
                # 실제 처리 자리: time.sleep은 예시용(실코드에선 처리 로직)
                time.sleep(0.02)
                if files:
                    self.message.emit(u"처리 중: {}".format(path))
                progress = int((float(i + 1) / total) * 100)
                self.progressed.emit(progress)

            self.message.emit("작업 완료")
        except Exception as e:
            self.message.emit("에러: {}".format(e))
        finally:
            self.finished.emit()

# ---------------- Main Window (GUI) ----------------
class Win(QtGui.QWidget):
    def __init__(self):
        super(Win, self).__init__()
        self.setWindowTitle("폴더 선택 & 진행상황 표시 (PyQt4)")
        v = QtGui.QVBoxLayout(self)

        # 폴더 선택 UI
        h = QtGui.QHBoxLayout()
        self.le_dir = QtGui.QLineEdit(); self.le_dir.setReadOnly(True)
        self.btn_pick = QtGui.QPushButton(u"폴더 선택…")
        h.addWidget(self.le_dir); h.addWidget(self.btn_pick)
        v.addLayout(h)

        # 시작/진행 UI
        self.btn_start = QtGui.QPushButton(u"작업 시작")
        self.pb = QtGui.QProgressBar(); self.pb.setRange(0, 100)
        v.addWidget(self.btn_start)
        v.addWidget(self.pb)

        # 로그
        self.log = QtGui.QPlainTextEdit(); self.log.setReadOnly(True)
        v.addWidget(self.log)

        # 상태
        self.dir_path = ""

        # 연결
        self.btn_pick.clicked.connect(self.on_pick_dir)
        self.btn_start.clicked.connect(self.on_start)

        self.thread = None
        self.worker = None

    # ----------- 슬롯들 -----------
    @QtCore.pyqtSlot()
    def on_pick_dir(self):
        # 1) 탐색기(폴더 선택) → GUI에 표시
        path = QtGui.QFileDialog.getExistingDirectory(
            self,
            u"폴더 선택",
            "",  # 시작 경로 (비워두면 최근 경로)
            QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
        )
        if path:
            # PyQt4 환경에 따라 QString일 수 있으니 str()로 보정
            self.dir_path = str(path)
            self.le_dir.setText(self.dir_path)
            self.append_log(u"[선택] {}".format(self.dir_path))

    @QtCore.pyqtSlot()
    def on_start(self):
        # 2) 내부 메인 함수(긴 작업) → QThread + 시그널로 진행상황 GUI 출력
        if self.thread is not None:
            return  # 이미 실행 중이면 무시(간단 보호)
        self.pb.setValue(0)
        self.append_log("작업을 시작합니다…")

        self.thread = QtCore.QThread(self)
        self.worker = Worker(self.dir_path)
        self.worker.moveToThread(self.thread)

        # 스레드 시작 시 워커 run 호출
        self.thread.started.connect(self.worker.run)

        # 진행/메시지/완료 연결
        self.worker.progressed.connect(self.pb.setValue)
        self.worker.message.connect(self.append_log)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_pick.setEnabled(False)

    @QtCore.pyqtSlot()
    def on_thread_finished(self):
        self.append_log("스레드 종료")
        self.thread = None
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_pick.setEnabled(True)

    # ----------- 유틸 -----------
    @QtCore.pyqtSlot(str)
    def append_log(self, msg):
        self.log.appendPlainText(msg)

# ---------------- Entrypoint ----------------
def main():
    app = QtGui.QApplication(sys.argv)
    w = Win(); w.resize(640, 420); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    
    
    
    
# PyQt5 기준 (PyQt6/PySide면 import만 바꾸면 됨)
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Win(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("폴더 선택 & 저장")
        self.links = {}  # 여기다 저장됨: {"data_dir": "C:/..."}
        
        self.btn = QtWidgets.QPushButton("폴더 선택")
        self.le  = QtWidgets.QLineEdit(); self.le.setReadOnly(True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(self.btn); lay.addWidget(self.le)

        self.btn.clicked.connect(lambda: self.pick_dir_and_save(key="data_dir"))

    @QtCore.pyqtSlot()
    def pick_dir_and_save(self, key="data_dir"):
        path = QFileDialog.getExistingDirectory(
            self,
            "폴더 선택",
            "",  # 시작 경로 (비워두면 최근)
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if path:  # 사용자가 선택했을 때만
            self.links[key] = path
            self.le.setText(path)
            print("saved:", key, "->", self.links[key])

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Win(); w.resize(520, 60); w.show()
    sys.exit(app.exec_())