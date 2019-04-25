## Document Clustering and Visualization
Github repo for CSE 573 project.
Contributors and Team Members: 
Kunal Suthar
Jay Shah 
Leroy Vargis
Abhay Mathur
Vatsal Sodha

### Selenium setup instructions:
1. Install selenium python package:
    ```sh
    pip install selenium
    ```
1. Install selenium browser driver:
    This project uses the [Firefox driver](https://github.com/mozilla/geckodriver/releases) Install instruction found [here](https://askubuntu.com/a/928514)

### Project run instructions:
To run the project, use demo.sh file with the following arguments:
1. Scrape data and save it to a pickle file:
    ```sh
    bash demo.sh scrape
    ```
2. Create the LDA model:
       ```sh 
        bash demo.sh create-lda
        ```
3. Apply tsne and generate the dependencies:
       ```sh 
        bash demo.sh apply-tsne
       ```
4. Apply pca and generate the dependencies:
       ```sh 
        bash demo.sh apply-pca
       ```
5. Generate 3D visualization:
       ```sh 
        bash demo.sh visualize-3d
       ```
6. Run the above steps/commands with tsne sequentially from scratch
```sh 
        bash demo.sh run-project-tsne
```
7. Run the above steps/commands with pca sequentially from scratch
```sh 
        bash demo.sh run-project-pca
```
