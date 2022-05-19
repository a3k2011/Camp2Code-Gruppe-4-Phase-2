# Camp2Code_Project_2

## PiCar:
Unter dem Einsatz eines Raspberry Pi Model 4 und dem Modelauto Bausatz Sunfounder PiCar-S, sowie in diesem Repository zur Verfügung gestellten Python-Code
kann das Modellauto betrieben werden.

#### Ausführung des Programms
Das Modellauto kann über ein Plotly-Dashboard im Webbrowser oder direkt in der Konsole angesteuert werden.
* PiCar_Dashboard.py
* PiCar.py

#### Anleitung zum Sunfounder PiCar-S
https://github.com/sunfounder/SunFounder_PiCar-S

#### Funktionen des PiCars
Das hier bereitgestellte Programm zum PiCar verfügt über die folgenden Funktionen:
#### Projektphase 1:
* Kalibrierung der IR-Sensoren
* Ausgabe der Messwerte der IR-Sensoren
* Fahrparcour 1-7
* Manuelle Steuerung des PiCars über das Plotly-Dashboard im Webbrowser
#### Projektphase 2:
* ...

## Installation notwendiger Software auf dem RP4
### OS
Raspberry Pi Desktop
* Debian Buster with Raspberry Pi Desktop
* https://downloads.raspberrypi.org/rpd_x86/images/
* Prüfe die installierte Version: cat /etc/*release
### Remote
VNC auf dem Raspberry Pi 4 aktivieren. Session: "Xorg"
* sudo apt-get install xrdp
* sudo systemctl status xrdp
### Einstellungen
Einstellungen -> Raspberry-Pi-Konfiguration -> Schnittstellen
* Kamera
* SSH
* VNC
* SPI
* I2C
### Allgemeines Projektphase 1
* sudo apt-get update
* sudo apt-get upgrade
  * sudo reboot
* sudo apt-get install python-smbus
* sudo apt-get install python3
* sudo apt-get install libatlas-base-dev
  * sudo reboot
### Allgemeines Projektphase 2
* sudo apt‑get install build‑essential cmake pkg‑config
* sudo apt‑get install libavcodec‑dev libavformat‑dev libswscale‑dev libv4l‑dev
* sudo apt‑get install libxvidcore‑dev libx264‑dev
* sudo apt‑get install libfontconfig1‑dev libcairo2‑dev
* sudo apt‑get install libgdk‑pixbuf2.0‑dev libpango1.0‑dev
* sudo apt‑get install libgtk2.0‑dev libgtk‑3‑dev
* sudo apt‑get install libatlas‑base‑dev gfortran
* sudo apt‑get install libhdf5‑dev libhdf5‑serial‑dev libhdf5‑103
* sudo apt‑get install libqtgui4 libqtwebkit4 libqt4‑test python3‑pyqt5 libjpeg8‑dev
* sudo apt‑get install libtiff5‑dev
* sudo apt‑get install libjasper‑dev libpng12‑dev libavcodec‑dev libavformat‑dev libswscale‑dev
libv4l‑dev
* sudo apt‑get install python3‑dev
  * sudo reboot
### Modul-Verwaltung
#### Requirements.txt
Alle notwendigen Bibliotheken können entweder manuell installiert werden (Siehe nächster Abschnitt Python-Module) oder über die Requirements.txt Datei installiert werden.
* pip3 install -r requirements.txt
#### Python-Module
* pip3 install --upgrade pip
* pip3 install numpy
* pip3 install pandas
* pip3 install plotly
* pip3 install dash pip3 install dash‑extensions
* pip3 install dash_daq
* pip3 install ‑U scikit‑learn
* pip3 install imgaug imgaug
* pip3 install https://github.com/lhelontra/tensorflow‑on‑arm/releases/download/v2.4.0/tensorflow‑2.4.0‑cp37‑none‑linux_armv7l.whl
 * sudo reboot
### OpenCV 4.5.5
* free -m
* wget https://github.com/Qengineering/Install‑OpenCV‑Raspberry‑Pi‑32‑bits/raw/main/OpenCV‑4‑5‑5.sh
* sudo chmod 755 ./OpenCV‑4‑5‑5.sh
* ./OpenCV‑4‑5‑5.sh
  * sudo reboot
## Fahrparcours
#### FP1 - Vorwärts und Rückwärts
Das Auto fährt mit langsamer Geschwindigkeit 3 Sekunden geradeaus, stoppt für 1 Sekunde und fährt 3 Sekunden rückwärts.
#### FP2 - Kreisfahrt mit maximalem Lenkwinkel
Das Auto fährt 1 Sekunde geradeaus, dann für 8 Sekunden mit maximalem Lenkwinkel im Uhrzeigersinn und stoppt.
Dann soll das Auto diesen Fahrplan in umgekehrter Weise abfahren und an den Ausgangspunkt zurückkehren.
Die Vorgehensweise soll für eine Fahrt im entgegengesetzen Uhrzeigersinn wiederholt werden.
#### FP3 - Vorwärtsfahrt bis Hindernis
Fahren bis ein Hindernis im Weg ist und dann stoppen. Während dieser Fahrt sollen die Fahrdaten 
(Geschwindigkeit, Lenkwinkel, Fahrtrichtung, Sensordaten) aufgezeichnet werden.
#### FP4 - Erkundungstour
Das Auto soll geradeaus fahren und im Falle eines Hindernisses die Fahrtrichtung ändern und die Fahrt fortsetzen.
Zur Änderung der Fahrrichtung soll ein maximaler Lenkwinkel eingeschlagen und rückwärts gefahren werden. Optional
können sowohl die Geschwindigkeit als auch die Fahrtrichtung bei freier Fahrt variiert werden. Zusätzlich sollen die 
Fahrdaten aufgezeichnet werden.
#### FP5 - Linienverfolgung
Folgen einer etwa 1,5 - 2 cm breiten Linie auf dem Boden. Das Auto soll stoppen, sobald das Auto das Ende der Linie erreicht hat.
Als Test soll eine Linie genutzt werden, die sowohl eine Rechts- als auch eine Linkskurve macht. Die Kurvenradien sollen deutlich 
größer sein als der maximale Radius, den das Auto ohne ausgleichende Fahrmanöver fahren kann.
#### FP6 - Erweiterte Linienverfolgung
Folgen einer Linie, die sowohl eine Rechts- als auch eine Linkskurve macht mit Kurvenradien kleiner als der maximale Lenkwinkel.
#### FP7 - Erweiterte Linienverfolgung mit Hindernisserkennung
Kombination von Linienverfolgung per Infrarot-Sensor und Hinderniserkennung per Ultraschall-Sensor.
Das Auto soll einer Linie folgen bis ein Hindernis erkannt wird und dann anhalten.

## GIT-Wiki:
#### Klonen eines vorhandenen Repositorys
* git clone git@github.com:a3k2011/Camp2Code-Gruppe-4-Phase-1.git
* git clone git@github.com:a3k2011/Camp2Code-Gruppe-4-Phase-2.git

#### Prüfung auf geänderte Dateien im eigenen Arbeitsverzeichnis
* git status

#### Auflistung aller durchgeführten push-Vorgänge
* git log

#### Arbeitsverzeichnis auf den aktuellen Status bringen
* git pull

#### Arbeitsverzeichnis zu GIT-Hub hochladen
1. git add --all oder (git add beispiel.txt)
2. git commit -m "Hier steht das Kommentar"
3. git push

## GIT-Einrichtung
* ssh-keygen -o (Beide Abfragen leer bestätigen.)
* cat /home/pi/.ssh/id_rsa.pub
* SSH-Key in GitHub einfügen

#### GIT-Config prüfen
* git config --list

## .gitignore
Ignoriert im Arbeitsverzeichnis die folgenden Dateien bzw. Ordner:
* \*.json
* /Logger
* /\__pycache__

## SD-Karte klonen / Backup
Tool: Win32 Disk Imager
1. Backup erstellen, indem *.img-Datei von altem Datenträger erstellt wird.
2. Backup lesen, indem *.img-Datei auf neuen Datenträger geschrieben wird
3. Partition auf Raspberry Pi erweitern, mit folgendem Befehl:
* sudo raspi-config --expand-rootfs
