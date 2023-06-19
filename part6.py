def print_story():
    # print story if exists
    try:
        with open('story.txt', 'r') as f:
            line = f.readline()

            while line:
                print(line)

                line = f.readline()

        print()
    except:
        print("story.txt does not exist.")