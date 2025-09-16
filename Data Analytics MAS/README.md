# Data Analytics MAS
This Multi-Agentic System (MAS) will be a data analytics system. Where the full workflow is to take in a csv file, apply cleaning, visualizations, and a report on NBA.

### Agents 
- Data Loader -> loads the dataset
- Analyst Agent -> computes the numerical stats on the data, and any other important values it seems
- Visualization Agent -> produces graphs/visualizations and stores in a folder names   `./graphs`
- Supervisor Agent -> stitches all the outputs together (data analysis + graphs) into a Markdown for a report

The agentic workflow is linear
$$START -> DataLoader -> Analyst -> Visualization -> Supervisor -> END$$

