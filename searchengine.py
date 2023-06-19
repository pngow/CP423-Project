import part1, part2, part3, part4, part5, part6

if __name__ == '__main__':
    option = None
    topics = None

    # while user hasn't exited
    while option != '7':
        print("""Select an option:
        1- Collect new documents.
        2- Index documents.
        3- Search for a query.
        4- Train ML classifier.
        5- Predict a link.
        6- Your story!
        7- Exit
    """)
        # get user input
        option = input()
        
        if option == '1':
            wc = part1.WebCrawler()
            wc.run()
            topics = wc.get_topics()
        elif option == '2':
            # need to run option 1 before selecting option 2 ... need mapping.txt and documents from crawl
            if topics is None:
                print('Please select option 1 first. No documents recently crawled. Index should be up to date.')
            else:
                part2.inverted_index(topics)
        elif option == '3':
            part3.search_query()
        elif option == '4':
            part4.train_classifier()
        elif option == '5':
            part5.run()
        elif option == '6':
            part6.print_story()
        elif option == '7':
            print("Exiting program.")
        else:
            print("Invalid option. Please enter the numeric value corresponding \
                to one of the above options.")