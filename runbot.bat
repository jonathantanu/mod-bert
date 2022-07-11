@echo off
if exist ".\env\Scripts\activate" (
		call  ".\env\Scripts\activate"
	) else (
		echo 'using global dependencies. Program will close if global doesn't found the required dependencies'
		pause
	)
python bot.py
pause
