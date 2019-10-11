## Installation of Walabot

Detailed Tutorial: https://learn.sparkfun.com/tutorials/getting-started-with-walabot/all 

GitHub page of Walabot: https://github.com/Walabot-Projects

### Linux installation

1. Download Linux [SDK](https://walabot.com/getting-started)

2. ` cd `to Downloads and 
` sudo dpkg -i WalabotSDK_2018.08.26_v1.2.2_Linux64.deb `

3. Agree to User License Agreement

4. pip install WalabotAPI.py library 
` pip install WalabotAPI --no-index --find-links="/usr/share/walabot/python/" `

5. Test example codes
` cd /usr/share/doc/walabot/examples/python `
` python *file*.py `


### Windows installation 

1. Download Windows [SDK](https://walabot.com/getting-started)

2. run WalabotSDK_2018.08.26_v1.2.2.exe as administrator and agree to everything

3. pip install WalabotAPI.py library 
` pip install WalabotAPI --no-index --find-links="%PROGRAMFILES%\Walabot\WalabotSDK\python\\" `

4. Connect walabot and test first with WalabotAPItutorial

5. Test example codes in
` cd C:\Program Files\Walabot\WalabotSDK\example\python `

## walabot scripts

The script initailizes and calibrates walabot. The following 
1. Use of  Tracker Profil
2. Use of RawSlicedImage() to receive raw 2D Image data of walabot