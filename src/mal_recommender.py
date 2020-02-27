import pandas as pd
import time

from jikanpy import Jikan
from surprise.dump import load

jikan = Jikan()

class MALRecommender(object):

	def __init__(self, algo, usernames, anime):
		super().__init__()
		self.algo = algo
		self.usernames = usernames # dictionary
		self.anime = anime # DataFrame
		self.users = pd.DataFrame(usernames).T.reset_index(
		)[["index"]].rename(columns={"index": "username"})

	def _get_uid(self, username):
		username = username.lower()
		if username in self.usernames:
			return self.usernames[username][0]
		else:
			return username

	def _fix_image_url(self, url):
		try:
			return "net".join(url.split("cdn-dena.com"))
		except:
			return "https://cdn.myanimelist.net/images/error/404_image.png"

	def get_recommendations(self, user, num_titles=10, filter_completed=True):
		"""
		Parameters
		----------
		user : str

		num_titles : int

		filter_completed : bool

		Returns
		-------
		recommendations : DataFrame

		"""
		user = self._get_uid(user)
		user_predictions = self.anime.reset_index()[["anime_id"]]
		if filter_completed:
			user_completed = pd.DataFrame(jikan.user(
				username=user, request='animelist', argument='completed', page=1)["anime"])
			i = 2
			downloading = True
			while downloading:
				time.sleep(4)  # API requires 4 second delay between requests
				downloading = jikan.user(
						username=user, request='animelist', argument='completed', page=i)["anime"]
				user_completed = user_completed.append(pd.DataFrame(downloading))
				i += 1
			user_completed_lst = list(user_completed["mal_id"])
			user_predictions["est"] = user_predictions["anime_id"].apply(
				lambda item: self.algo.predict(user, item)[3] if item not in user_completed_lst else 0)
		else:
			user_predictions["est"] = user_predictions["anime_id"].apply(
				lambda item: self.algo.predict(user, item)[3])
		return user_predictions.sort_values("est", ascending=False).head(num_titles)["anime_id"].apply(lambda x: self.anime.loc[x])


	def get_fans(self, item, num_users=10, get_estimates=True):
		"""
		Parameters
		----------
		item : int

		num_users : int

		get_estimates : bool

		Returns
		-------
		fans : DataFrame

		"""
		item_predictions = self.users.copy()
		item_predictions["est"] = item_predictions["username"].apply(
				lambda user: self.algo.predict(self._get_uid(user), item)[3])
		if get_estimates:
			return item_predictions.sort_values("est", ascending=False).head(num_users)
		else:
			return item_predictions.sort_values("est", ascending=False).head(num_users)[["username"]]
