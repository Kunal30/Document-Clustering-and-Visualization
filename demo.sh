#!/bin/bash
if [ "$1" == "install-dep" ]; then
	echo 'Installing dependencies....'
	sudo pip install gensim
	sudo pip install plotly
	sudo pip install numpy
	sudo pip install sklearn
	sudo pip install matplotlib

elif [ "$1" == "scrape" ]; then
	echo 'Scraping data from the mentioned website....'
	python Code/scraper/cnn_scraper_selenium.py

elif [ "$1" == "create-lda" ]; then
	echo 'BUILDING UP THE LDA Model....'
	python Code/src/getLDAModel.py

elif [ "$1" == "apply-tsne" ]; then
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python Code/src/lda2tsne.py

elif [ "$1" == "apply-pca" ]; then
	echo 'Applying PCA on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python Code/src/lda2pca.py

elif [ "$1" == "visualize-3d" ]; then
	echo 'Setting up environment for 3D visualization........'
	python Code/src/3D_Visualization.py

elif [ "$1" == "run-project-tsne" ]; then
	echo 'Running the whole project from the beginning.......'
	echo 'BUILDING UP THE LDA Model....'
	# python getLDAModel.py
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python Code/src/lda2tsne.py
	echo 'Setting up environment for 3D visualization........'
	python Code/src/3D_Visualization.py

elif [ "$1" == "run-project-pca" ]; then
	echo 'Running the whole project from the beginning.......'
	echo 'BUILDING UP THE LDA Model....'
	python Code/src/getLDAModel.py
	echo 'Applying PCA on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python Code/src/lda2pca.py
	echo 'Setting up environment for 3D visualization........'
	python Code/src/3D_Visualization.py


else
	echo "use \"install-dep\" argument to install dependencies"
	echo "use \"scrape\" argument to scrape data and save it to a pickle file"
	echo "use \"create-lda\" argument to create the lda model"
	echo "use \"apply-tsne\" argument to  apply tsne and generate the dependencies"
	echo "use \"apply-pca\" argument to  apply pca and generate the dependencies"
	echo "use \"visualize-3d\" argument to generate 3D visualization"
	echo "use \"run-project-tsne\" argument to run the above 3 steps/commands with tsne sequentially from scratch"
	echo "use \"run-project-pca\" argument to run the above 3 steps/commands with pca sequentially from scratch"
fi	
	