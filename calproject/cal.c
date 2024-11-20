#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Array of month names
char *months[] = {"January", "February", "March", "April", "May", 
                    "June", "July", "August", "September", "October", "November", "December"};

// Array of days in each month (non-leap year)
int dayList[] = {31, 28, 31, 30, 31, 
                    30, 31, 31, 30, 31, 30, 31};

// Checks if the first 3 letters of the input match any month.
int compareMonths(char *input){
    for (int i = 0; i < 12; i++){
        // Compare the first 3 letters using strncasecmp(for case)
        if (strncasecmp(input, months[i], 3) == 0){
            return i + 1; 
        }
    }
    return 0; 
}

// Determines the length of the year
int yearLength(int year){

    int yearL = 0;
    if (year < 10) {
        yearL = 1;
    }
    else if (year < 100){
        yearL = 2;
    }
    else if (year < 1000){
        yearL = 3;
    }
    else if (year < 10000){
        yearL = 4; 
    }
    else{
        yearL = 0;
    }

    return yearL;
}

// Checks if a year is a leap year
int isLeapYear(int year){
    if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)) 
        return 1;
    else 
        return 0;
}

// Calculates the day of the week for a given date
int date(int month, int day, int year){
    
    int tempDayList[12] = {31, 28, 31, 30, 31, 30, 
                           31, 31, 30, 31, 30, 31};

    // Adjust for leap year
    if(isLeapYear(year)){
        tempDayList[1] = 29;
    }

    int acc_days = 0;
    for(int i = 1753; i < year; i++){
        if (isLeapYear(i) == 1) {
            acc_days = acc_days + 366;
        }
        else {
            acc_days = acc_days + 365;
        }
    }

    for(int i = 0; i < month-1; i++){
        acc_days += tempDayList[i];
    }

    acc_days += day;
    int day_of_week = acc_days % 7;
    return day_of_week;
}

// Finds the current month
int findCurrentMonth(){
    time_t timenow;
    struct tm *current;
    time (&timenow);
    current = localtime (&timenow);
    return current->tm_mon + 1;
}

// Finds the current year
int findCurrentYear(){
    time_t timenow;
    struct tm *current;
    time (&timenow);
    current = localtime (&timenow);
    return current->tm_year + 1900;
}

// Prints the month calendar
int printMonth(int month, int year){
    char *mon = months[month-1];
    int length = strlen(mon);
    int length2 = yearLength(year);
    int spaces = (20 - (length + length2)) / 2;
    for(int i = 0; i < spaces; i++){
        printf(" ");
    }

    printf("%s %d\n", mon, year);
    printf("Su Mo Tu We Th Fr Sa\n");

    // Determine the starting day of the week
    int start_day = date(month, 1, year); // Get the day of the week for the 1st

    // Print initial spaces
    for(int i = 0; i < start_day; i++) {
        printf("   ");
    }

    // Determine the number of days in the month
    int days_in_month = dayList[month-1];
    if(month == 2 && isLeapYear(year)){
        days_in_month = 29;
    }

    for(int i = 1; i <= days_in_month; i++){
        printf("%2d ", i); // Always aligned 2 spaces to the right 
        if( (i + start_day) % 7 == 0){
            printf("\n");
        }
    }
    printf("\n");
    return 0;
}

// Prints the entire year
int printYear(int year){
    for(int i = 1; i <= 12; i++){
        printMonth(i, year);
        printf("\n");
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // Parse inputs in main

    if (argc == 1){
        // Call function with current month and year
        int month = findCurrentMonth();
        int year = findCurrentYear();
        printMonth(month, year);
        printf("\n");
    }
    else if (argc == 2){
        // Print the entire year
        int year = atoi(argv[1]);
        if (year >= 1753 && year <= 9999){
            printYear(year);
        }
        else {
            printf("Erhm, try entering a proper year.\n");
            return 1; 
        }
    }
    else if (argc == 3){
        if (strcmp(argv[1], "-m") == 0){
            // Handle month associated with -m 
            // Try to interpret argv[2] as a number and then as -m
            int month = atoi(argv[2]);
            if (month >= 1 && month <= 12){
                int year = findCurrentYear();
                printMonth(month, year);
            }
            else{
                month = compareMonths(argv[2]);
                if (month > 0){
                    int year = findCurrentYear();
                    printMonth(month, year);
                }
                else{
                    printf("Erhm, seems to be an invalid month/number.\n");
                    return 1; 
                }
            }
        }
        else{
            printf("Erhm, error.\n");
            return 1; 
        }
    }
    else if (argc == 4){
        if (strcmp(argv[1], "-m") == 0){
            int year = atoi(argv[3]);
            if (year < 1753 || year > 9999){
                printf("Enter a proper year.\n");
                return 1; 
            }

            // Parse month
            int month = atoi(argv[2]);
            if (month >= 1 && month <= 12){
                printMonth(month, year);
            }
            else{
            
                month = compareMonths(argv[2]);
                if (month > 0){
                    printMonth(month, year);
                }
                else{
                    printf("Erhm, invlaid month name/number.\n");
                    return 1; 
                }
            }
        }
        else{
            printf(";[ error.\n");
            return 1;
        }
    }
    else{
        printf("Invalid input format.\n");
        return 1; 
    }

    return 0;
}
