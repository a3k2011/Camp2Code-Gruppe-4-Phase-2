import os, json, time
from datetime import datetime


class Datenlogger:
    """Datenlogger Klasse

    Funktion:
    Speichert uebergebene Tupels oder Listen mit Angabe des Zeitdeltas ab Start der Aufzeichnung in ein JSON-File.

    Returns:
        [*.json]: Messwerte aus uebergebenen Daten mit bliebigem Zeitinterval.
    """

    def __init__(self, log_file_path=None):
        """Zielverzeichnis fuer Logfiles kann beim Init mit uebergeben werden.
            Wenn der Ordner nicht existiert wird er erzeugt.

        Args:
            log_file_path (_type_, optional): Angabe des Zielordners. Defaults to None.
        """
        self._log_file = {}
        self._log_data = []
        self._start_timestamp = 0
        self._logger_running = False
        self._log_file_path = log_file_path

    def start(self):
        """Funktion startet den Datenlogger."""

        self._logger_running = True
        self._start_timestamp = time.time()
        self._log_file["start"] = str(datetime.now()).partition(".")[0]

    def append(self, data):
        """Funktionen fuegt Daten in die Liste des Datenloggers hinzu."""

        if self._logger_running:
            ts = round((time.time() - self._start_timestamp), 2)
            self._log_data.append([ts] + data)

    def save(self):
        """Funktion speichert die uebergebenen Daten."""

        if self._logger_running and (len(self._log_data) > 0):
            self._logger_running = False
            self._log_file["data"] = self._log_data
            self._log_file["ende"] = str(datetime.now()).partition(".")[0]
            filename = self._log_file.get("start").partition(".")[0]
            filename = (
                filename.replace("-", "").replace(":", "").replace(" ", "_")
                + "_drive.log"
            )
            if self._log_file_path != None:
                logfile = os.path.join(self._log_file_path, filename)
                if not os.path.isdir(self._log_file_path):
                    os.mkdir(self._log_file_path)
            else:
                logfile = filename
            with open(logfile, "w") as f:
                json.dump(self._log_data, f)
            self._log_file.clear()
            self._log_data.clear()
            print("Log-File saved to:", logfile)