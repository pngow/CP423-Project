# CP423-Project

The objective of this project was to create a topical search engine which would allow users to do the following:
  1. Collect new documents related to the topics focused on by the search engine by crawling a list of source URLs for each topic to find more links (100 per topic)
  2. Index the collected documents with an inverted index, listing all the terms that show up in the documents and which document they appear in
  3. Create a search query to retrieve the 3 most relevant documents that have at least one word from that query in the search engine using the term at a time algorithm
  4. Train a multinomial Naive Bayes classifier to predict an unseen URL's topic category from those that the search engine focuses on
  5. Predict the topic category of the URL inputted by the user by crawling, extracting, processing/cleaning, and vectorizing it using the Naive Bayes classifier trained in part 4
  6. Exit the search engine

Users are prompted to select one of the options when they run the program.
