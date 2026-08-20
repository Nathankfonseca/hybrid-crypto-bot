import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    fig, ax = plt.subplots()
    plot_time = ["10:00:00", "10:00:03", "10:00:06"]
    plot_port = [1000, 1005, 1010]
    ax.plot(plot_time, plot_port)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    fig.canvas.draw()
    print("Plot SUCCESS")
except Exception as e:
    print("Plot ERROR:", e)
