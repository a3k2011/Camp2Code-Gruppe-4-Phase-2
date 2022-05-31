# Camp2Code_Project_2

## PiCar:
Unter dem Einsatz eines Raspberry Pi Model 4 und dem Modelauto Bausatz Sunfounder PiCar-S, sowie in diesem Repository zur Verfügung gestellten Python-Code
kann das Modellauto betrieben werden.

#### Ausführung des Programms
Das Modellauto kann über ein Plotly-Dashboard im Webbrowser angesteuert werden.
* CamCar_Dashboard.py

#### Anleitung zum Sunfounder PiCar-S
https://github.com/sunfounder/SunFounder_PiCar-S

#### Funktionen des PiCars
Das hier bereitgestellte Programm zum PiCar verfügt über die folgenden Funktionen:
* Fahrparcour mit OpenCV
* Fahrparcour mit DeepNN
* Manuelle Steuerung des PiCars über das Plotly-Dashboard im Webbrowser

Das Plotly-Dashboard verfügt über die folgenden Funktionen:
* Live-View der PiCar-Kamera
* Ausführung der Fahrparcours
* Parameter-Tuning zur Bildvorverarbeitung
* Speicherung der Parameter in die config.json
* Visualisierung der KPI's
* Visualisierung der Fahrdaten aus dem Datenlogger

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
* pip3 install dash
* pip3 install dash-extensions==0.0.70
* pip3 install dash_daq
* pip3 install ‑U scikit‑learn
* pip3 install --no-binary imgaug imgaug
* pip3 install https://github.com/lhelontra/tensorflow‑on‑arm/releases/download/v2.4.0/tensorflow‑2.4.0‑cp37‑none‑linux_armv7l.whl
  * sudo reboot
### OpenCV 4.5.5
* free -m
* wget https://github.com/Qengineering/Install‑OpenCV‑Raspberry‑Pi‑32‑bits/raw/main/OpenCV‑4‑5‑5.sh
* sudo chmod 755 ./OpenCV‑4‑5‑5.sh
* ./OpenCV‑4‑5‑5.sh
  * sudo reboot
## Fahrparcours
#### FP1 - Parameter-Tuning
Das Auto ist im Stillstand. Anhand der Live-View können Canny Edge Detection und HoughLinesP aktiviert werden. Alle zur Verfügung gestellten Parameter können live angepasst und in die config.json gespeichert werden.
#### FP2 - OpenCV
Das Auto fährt mit der im Slider eingestellten Geschwindigkeit und nutzt die OpenCV Lane Detection auf Basis der definierten Parameter, um das PiCar durch den Fahrparcour zu lenken.
#### FP2 - OpenCV
Das Auto fährt mit der im Slider eingestellten Geschwindigkeit und nutzt die DeepNN Lane Detection (cnn_model.h5), um das PiCar durch den Fahrparcour zu lenken.## GIT-Wiki:
### GIT-Einrichtung (RP4)
#### SSH erzeugen
* ssh-keygen -o (Beide Abfragen leer bestätigen, ausser man möchte auch ein Passwort vergeben.)
  * Erzeugt den SSH-Key.
* cat /home/pi/.ssh/id_rsa.pub
  * Gibt den SSH-Key im Terminal aus. 
* SSH-Key in GitHub registrieren
  * In GitHub einloggen und in die Account-Settings unter SSH and GBP Keys navigieren
  * Titel: Bezeichnung des verwendeten Geräts
  * Key: SSH-Key
#### Klonen eines vorhandenen Repositorys
* git clone git@github.com:a3k2011/Camp2Code-Gruppe-4-Phase-1.git
* git clone git@github.com:a3k2011/Camp2Code-Gruppe-4-Phase-2.git

#### GIT-Konfiguration prüfen
* cd REPOSITORYNAME
* GitHub-Username und E-Mail konfiguieren:
  * git config user.name "USERNAME"
  * git config user.email EMAIL
* git config --list
  * Prüfung der Konfiguration.

### Git-Kommandos
#### Prüfung auf geänderte Dateien im eigenen Arbeitsverzeichnis
* git status

#### Auflistung aller durchgeführten push-Vorgänge
* git log

#### Liste der verfolgten Dateien
* git ls-files

#### Arbeitsverzeichnis auf den aktuellen Status bringen
* git pull

#### Arbeitsverzeichnis zu GIT-Hub hochladen
1. git pull
2. git add --all oder (git add beispiel.txt)
3. git commit -m "Hier steht das Kommentar"
   * (Optional: Erweitern des letzten Commits)
   * git commit --amend -m "Das ist die richtige Message"
4. git push

#### Entfernen der Änderungen im Staging-Bereich
* git reset

### Commit rückgängig machen
* git reset HEAD~

## .gitignore
Ignoriert im Arbeitsverzeichnis die folgenden Dateien bzw. Ordner:
* \*.json
* \*.h5
* /Logger
* /IMG_Logger
* /\__pycache__

## SD-Karte klonen / Backup
Tool: Win32 Disk Imager
1. Backup erstellen, indem *.img-Datei von altem Datenträger erstellt wird.
2. Backup lesen, indem *.img-Datei auf neuen Datenträger geschrieben wird
3. Partition auf Raspberry Pi erweitern, mit folgendem Befehl:
* sudo raspi-config --expand-rootfs