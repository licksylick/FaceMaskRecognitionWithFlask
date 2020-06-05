# FaceMaskRecognitionWithFlask
Face-mask recognition project based on Python and Flask

Deploy on local machine:
1. Install necessary packages and libraries
2. Run app.py
3. Project will be available at http://127.0.0.1:4555/

Deploy on Heroku:
(you need paid plan)
1. Open project folder
2. Create requirements.txt file: pip freeze > requirements.txt
3. Use commands:
  heroku login
  heroku create projectname
  git init
  git add .
  git status
  git commit -m “my commit”
  heroku git:remote -a projectname
  git push heroku master

