# Automate Your Day Job with Python

## PyBank

![Revenue](Images/revenue-per-lead.jpg)

I created a Python script for analyzing the financial records of a company. I utilized a financial dataset in this file: [budget_data.csv](PyBank/Resources/budget_data.csv). This dataset is composed of two columns, Date and Profit/Losses.

I created a Python script that analyzes the records to calculate each of the following:

* The total number of months included in the dataset.

* The net total amount of Profit/Losses over the entire period.

* The average of the changes in Profit/Losses over the entire period.

* The greatest increase in profits (date and amount) over the entire period.

* The greatest decrease in losses (date and amount) over the entire period.

The resulting analysis looks similar to the following:

  ```text
  Financial Analysis
  ----------------------------
  Total Months: 86
  Total: $38382578
  Average  Change: $-2315.12
  Greatest Increase in Profits: Feb-2012 ($1926159)
  Greatest Decrease in Profits: Sep-2013 ($-2196167)
  ```

The final script prints the analysis to the terminal and exports a text file with the results.

## PyRamen (Optional)

![ramen.jpg](Images/ramen.jpg)

## Background

Welcome to Ichiban Ramen!

I created a script to analyze my ramen business's financial performance by cross-referencing my sales data with my internal menu data to figure out revenues and costs for the year.

I also want to analyze how well my business did on a per-product basis in order to better understand which products are doing well, which are doing poorly, and, ultimately, which products may need to be removed or changed.

This tool allows automation of my calculations in a manner that scales with my business.

---

### Contents

* Read in `menu_data.csv` and set its contents to a separate list object. (This way, you can cross-reference your menu data with your sales data as you read in your sales data in the coming steps.)

  * Initialize an empty `menu` list object to hold the contents of `menu_data.csv`.

  * Use a `with` statement and open the `menu_data.csv` by using its file path.

  * Use the `reader` function from the `csv` library to begin reading `menu_data.csv`.

  * Use the `next` function to skip the header (first row of the CSV).

  * Loop over the rest of the rows and append every row to the `menu` list object (the outcome will be a list of lists).

* Set up the same process to read in `sales_data.csv`. However, instead append every row of the sales data to a new `sales` list object.

* Initialize an empty `report` dictionary to hold the future aggregated per-product results. The `report` dictionary will eventually contain the following metrics:

  * `01-count`: the total quantity for each ramen type

  * `02-revenue`: the total revenue for each ramen type

  * `03-cogs`: the total cost of goods sold for each ramen type

  * `04-profit`: the total profit for each ramen type

* Then, loop through every row in the `sales` list object.

  * For each row of the `sales` data, set the following columns of the sales data to their own variables:

    * Quantity
    * Menu_Item

  * Perform a quick check if the `sales_item` is already included in the `report`. If not, initialize the key-value pairs for the particular `sales_item` in the report. Then, set the `sales_item` as a new key to the `report` dictionary and the values as a nested dictionary containing the following:

    ```python
    {
    "01-count": 0,
    "02-revenue": 0,
    "03-cogs": 0,
    "04-profit": 0,
    }
    ```

* Create a nested loop by looping through every record in `menu`.

  * For each row of the `menu` data, set the following columns of the menu data to their own variables:

    * Item
    * Price
    * Cost

  * If the `sales_item` in sales is equal to the `item` in `menu`, capture the `quantity` from the sales data and the `price` and `cost` from the menu data to calculate the `profit` for each item.

    * Cumulatively add the values to the corresponding metrics in the report like so:

      ```python
      report[sales_item]["01-count"] += quantity
      report[sales_item]["02-revenue"] += price * quantity
      report[sales_item]["03-cogs"] += cost * quantity
      report[sales_item]["04-profit"] += profit * quantity
      ```

  * Else print the message "{sales_item} does not equal {item}! NO MATCH!".

* Write out the contents of the `report` dictionary to a text file. The report should output each ramen type as the keys and `01-count`, `02-revenue`, `03-cogs`, and `04-profit` metrics as the values for every ramen type.

---

## Resources

* [Stack Overflow](https://www.stackoverflow.com): A wealth of community-driven questions and answers, particularly effective for IT solution seekers.

* [Python Basics](https://pythonbasics.org/): Contains example materials and exercises for the Python 3 programming language.

* [Python Documentation](https://docs.python.org/3/): Official Python documentation

---
