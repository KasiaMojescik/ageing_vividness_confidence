# Publication: The Relationship Between Subjective Memory Experience and Objective Memory Performance Remains Stable Across the Lifespan
<a href="https://doi.org/10.1525/collabra.116195">Mojescik, K. M., Berens, S. C., De Luca, F., Ritchey, M., & Bird, C. M. (2024). The Relationship Between Subjective Memory Experience and Objective Memory Performance Remains Stable Across the Lifespan. Collabra: Psychology, 10(1).</a>

### Final version of the code and data linked in the publication can be found at Open Science Framework: https://osf.io/ceq86/
### Pre-registration: https://osf.io/5vcpt

## Abstract
The way humans remember events changes across the lifespan. Older adults often rate the vividness of their memories as being greater or equal to younger adults, despite poorer performance on episodic memory tasks. This study explored how the content (place, person and object) and specificity (conceptual gist versus perceptual detail) of event memories relate to the subjective experience of memory vividness and memory confidence, and how this relationship is affected by healthy ageing. 100 healthy older adults and 100 young adults were tested online, using an adapted version of a paradigm developed by Cooper and Ritchey (2022). At encoding, participants generated a distinctive story to associate together (1) a theme word, and images of (2) a famous person, (3) a place, and (4) an object, to create unique events. At test, participants identified the event components using word labels (indexing conceptual gist), and the studied images (indexing perceptual details). Replicating Cooper and Ritchey (2022), we found that young adults’ memory vividness ratings were related to their memory for the conceptual gist of the events, with no modulation by the type of the content recalled. Strikingly, older adults showed the same relationship between vividness measures and objective performance as the young adults. Contrary to some previous studies, we found that older adults obtained lower scores for gist-based memory, and their vividness ratings were correspondingly lower than the younger adults. Across both age groups, vividness and confidence ratings followed a similar pattern, showing a stronger relationship with conceptual gist. Our results suggest that throughout the lifespan, the amount of conceptual information retrieved about an event relates to the ability to reexperience it vividly, and to have confidence in one’s memory.

## Design
This experiment had a 2 (between: young vs older adults) x 2 (within: gist vs detail) x 3 (within: place vs object vs person) mixed-design. There were three dependent variables: memory performance, vividness rating, and confidence rating. 

## Code and Data
To replicate the results published in the paper, please use to the final version of the code and data published at Open Science Framework: https://osf.io/ceq86/. <br>
This repository served as a version control during the analysis development process and there may be slight changes in the code from the final published version. <br>
Please see main_analyses.ipynb and three_way_anova.html for the main analyses tackling the hypotheses, and individual_differences_analyses.ipynb for the K-Means clustering analyses reported in the publication. <br>See lmm_education.html/qmd for the demographic control analyses reported in Supplementary Materials accessible at https://osf.io/ceq86/. 

## Acknowledgement
Analyses in main_analyses.ipynb, helper_functions.py, clustering.py, kmeans_clustering.ipynb, individual_differences_analyses.ipynb, individual_differences_helper_functions.py are based on the code provided by Cooper and Ritchey (2022), which can be found here: https://github.com/memobc/paper-vividness-features.<br>
They have been extended to allow for an additional manipulation of an age group (both young and older adults have been tested) and an additional dependent variable - a confidence rating.

## Contact
For any questions or if you spot any issues, please contact k.mojescik@sussex.ac.uk
