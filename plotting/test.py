# To create plots, we first need to import matplotlib
# We name it plt when we import it, so it's easier to work with (we don't want to have to keep typing matplotlib.pyplot)
import matplotlib.pyplot as plt

# Now we can make a scatter plot.
# A scatter plot is a simple chart that shows pairs of values as dots with x and y axes.
# Let's say that we've taken random readings of temperature throughout the year, and we want to graph temperature vs month to see which months are colder
# Temperature is in fahrenheit, and is the temperature on a random day in the corresponding month
# The month at index 0 matches up with the temperate at index 0, and so on.
month = [0,1,2,2,4,5,5,7,8,10,10,11,12]
temperature = [32,15,40,35,50,55,52,80,85,60,57,45,35]
# We tell matplotlib to draw the scatterplot with this command.
plt.scatter(month, temperature)

# This command will show the drawn plot
# Look at the output area below.
# You'll see the plot drawn there.
plt.show()