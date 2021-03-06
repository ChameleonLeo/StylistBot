# NeuralStylist

Bot is avaliable by username @NeuralStylistBot<br><br>
NeuralStylist bot is powered by Heroku server using the idle concept (the app goes to sleep after 30 minutes 
of inactivity). Thus, it takes about a minute for the Bot to wake up. Please, be patient when contacting Bot.<br>
NeuralStylist designed as an asynchronous bot (when being deployed on server with enough RAM,
it can maintain concurrent requests). But it is not recomended to check current version of NeuralStylist
from few accounts simultaniously, because there are RAM restrictions on free Heroku account,
what can cause the crash of the process and the server will just restart.


*<b>An important fact:</b>
Model's performance intentionally decreased, because of RAM restrictions on free Heroku account.
The result would be not impressive (picture resolution is too low).*<br>
Bot was deployed just for functionality demonstration.
Locally (depending on device characteristics) it works much better and faster.

Examples of input and output images:
<br>
1. Still life by Vincent van Gogh
<div align="center">
<div class="input">
<img src="images/content/still_life.jpg" width="400" alt="content_example1">
<img src="images/style/van_gogh2.jpg" width="300" alt="style_example1">
</div>
<img src="images/results/photo5337057517882160029.jpg" width="400" alt="result_example1">
</div>
<br>
<br>
<br>
2. Still life by Edvard Munch
<div align="center">
<div class="input">
<img src="images/content/still_life1.jpg" width="400" alt="content_example2">
<img src="images/style/munch3.jpg" height="350" alt="style_example2">
</div>
<img src="images/results/photo5339309317695845068.jpg" width="400" alt="result_example2">
</div>
<br>
<br>
<br>
3. Landscape by Salvador Dali
<br>
<br>
<div align="center">
<div class="input">
<img src="images/content/landscape0.jpg" width="400" alt="content_example3">
<img src="images/style/dali0.jpg" width="400" alt="style_example3">
</div>
<img src="images/results/photo5339309317695845065.jpg" width="400" alt="result_example3">
</div>
<br>
<br>
<br>
4. Landscape by others
<br>
<br>
<div align="center">
<div class="input">
<img src="images/content/landscape0.jpg" width="400" alt="content_example4">
</div>
<img src="images/results/screenshot.PNG" alt="result_example4">
</div>
