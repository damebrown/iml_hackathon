from parsing import *


def hashtag_plot():
    avg_hashtags = []
    var_hashtags = []
    for i in range(10):
        tweets = np.array(get_tweets(paths[i])['tweet'])
        hashtags = []
        for tweet in tweets:
            hashtags.append(len(extract_hashtags(tweet)))
        avg_hashtags.append(np.mean(hashtags))
        var_hashtags.append(np.var(hashtags))
    plt.scatter(names, avg_hashtags, label="average num of hashtags in tweet")
    plt.scatter(names, var_hashtags, label="variance of hashtags")
    plt.title("average num of hashtags in tweets")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("average num of hashtags in tweets")
    plt.show()


def tags_plot():
    avg_tags = []
    var_tags = []
    for i in range(10):
        tweets = np.array(get_tweets(paths[i])['tweet'])
        tags = []
        for tweet in tweets:
            tags.append(len(extract_tags(tweet)))
        avg_tags.append(np.mean(tags))
        var_tags.append(np.var(tags))
    plt.scatter(names, avg_tags, label="average num of tags in tweet")
    plt.scatter(names, var_tags, label="variance of tags")
    plt.title("average num of tags in tweets")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("average num of tags in tweets")
    plt.show()


def emojis_plot():
    avg_em = []
    var_em = []
    for i in range(10):
        tweets = np.array(get_tweets(paths[i])['tweet'])
        e = []
        for tweet in tweets:
            e.append(count_emoji(tweet))
        avg_em.append(np.mean(e))
        var_em.append(np.var(e))
    plt.scatter(names, avg_em, label="average num of emojis in tweet")
    plt.scatter(names, var_em, label="variance of emojis")
    plt.title("average num of emojis in tweets")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("average num of emojis in tweets")
    plt.show()


def spanish_plot():
    avg_s = []
    var_s = []
    for i in range(10):
        tweets = np.array(get_tweets(paths[i])['tweet'])
        s = []
        for tweet in tweets:
            if is_it_spanish(tweet):
                s.append(1)
            else:
                s.append(0)
        avg_s.append(np.mean(s))
    plt.scatter(names, avg_s, label="average num of spanish word in tweet")
    plt.title("average num of spanish words in tweets")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("average num of spanish words in tweets")
    plt.show()


def link_plot():
    avg_s = []
    var_s = []
    for i in range(10):
        tweets = np.array(get_tweets(paths[i])['tweet'])
        s = []
        for tweet in tweets:
            s.append(count_link(tweet))
        avg_s.append(np.mean(s))
        var_s.append(np.var(s))
    plt.scatter(names, avg_s, label="average num of links in tweet")
    plt.scatter(names, var_s, label="variance of links in tweets")
    plt.title("average num of links in tweets")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("average num of links in tweets")
    plt.show()


spanish_plot()
hashtag_plot()
tags_plot()
emojis_plot()
link_plot()
