Subdirectories include instructions for setup.

npm install to setup JS libraries locally.
Instructions for backend libraries in subdirectory README.s

Frontend interface (view): `npm run start`
Backend implementation (model): `flask run` to load,
`flask db init`, `flask db migrate -m "migration note"`, `flask db upgrade` to set up SQLite3 db locally.

Adjust DB URL to be where your model folder is from the URL below:
DATABASE_URL="sqlite:////home/ali/individual-project-app/model/app.db"

Add this to a .env file in the model folder.