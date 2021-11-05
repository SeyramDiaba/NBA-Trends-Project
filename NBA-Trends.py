import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# Comparing the 'knicks' to 'nets' wrt points earned per game
knicks_pts = nba_2010.pts[nba.fran_id == 'Knicks']
nets_pts = nba_2010.pts[nba.fran_id == 'Nets']

# Difference between the two teams average points
diff_means_2010 = np.mean(knicks_pts) - np.mean(nets_pts)
print('2010 average points difference: ',diff_means_2010)


# Plot an overlapping histogram to compare points
plt.hist(knicks_pts, label='Knicks',normed = True,alpha = 0.8)
plt.hist(nets_pts, label = 'Nets', normed = True, alpha = 0.8)
plt.legend()
plt.show()
plt.clf()

# Comparing 2014 game points between knicks and nets
knicks_pts_2014 = nba_2014.pts[nba['fran_id'] == 'Knicks']
nets_pts_2014 = nba_2014.pts[nba['fran_id'] == 'Nets']

# Mean difference between knicks and nets in 2014
diff_means_2014 = np.mean(knicks_pts_2014) - np.mean(nets_pts_2014)
print('2014 average points difference: ', diff_means_2014)

# Investigate the relationship between franchise and points scored per game
sns.boxplot(data = nba_2010, x='fran_id', y='pts')
plt.show()
plt.clf()

# Contingency table between 'game_result' and 'game_location'
location_result_freq = pd.crosstab(nba_2010.game_result,nba_2010.game_location)
print(location_result_freq)

# Converting contingency table of frequency to a contingency table of proportions
location_result_proportions = location_result_freq/len(nba_2010)
print(location_result_proportions)

# Using contingency table created in previous code to calculate the expected contingency table if there were no association.
chi2,pval,dof, expected = chi2_contingency(location_result_proportions)
print(expected)
print('Chi-Square Statistic: ',chi2)

# Finding the covariance between forecast and point_diff
forpoint_cov = np.cov(nba_2010.forecast,nba_2010.point_diff)
print(forpoint_cov)
print('Covariance is: ', forpoint_cov[0][1])

#  Calculating the correlation between forecast and point difference
forpoint_corr, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print("Correlation is :", forpoint_corr) 

# Scatter plot of forecast on x-axis and point_diff on y=axis
plt.scatter(x = nba_2010['forecast'], y = nba_2010['point_diff'])
plt.xlabel('Forecast Win Probability')
plt.ylabel('Point Differential')
plt.show()
plt.clf()