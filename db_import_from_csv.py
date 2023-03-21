import csv
import sqlite3
import glob
import os

def do_directory(dirname, db):
    # This does the same thing to all the csvs in a directory specified in the "dirname"
    # field. We need to create a script that takes either csvs or db and concatenates them
    # in a single db for processing
    for filename in glob.glob(os.path.join(dirname, '*.csv')):
        do_file(filename, db)

def do_file(filename, db):
        with open(filename) as f:
            with db:
                data = csv.DictReader(f)
                cols = data.fieldnames
                table=os.path.splitext(os.path.basename(filename))[0]

                sql = 'drop table if exists "{}"'.format(table)
                db.execute(sql)

                sql = 'create table "{table}" ( {cols} )'.format(
                    table=table,
                    cols=','.join('"{}"'.format(col) for col in cols))
                db.execute(sql)

                sql = 'insert into "{table}" values ( {vals} )'.format(
                    table=table,
                    vals=','.join('?' for col in cols))
                db.executemany(sql, (list(map(row.get, cols)) for row in data))

if __name__ == '__main__':
    conn = sqlite3.connect('BEAIKL1_PRISM_DT_Tags_full_20220919.db')
    do_directory('C:\Expert System Codes\Data_transfer\Beware\TestDataDir', conn)
