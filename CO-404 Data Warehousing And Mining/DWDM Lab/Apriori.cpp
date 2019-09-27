#include <bits/stdc++.h>
#include <map>
using namespace std;

ifstream fin;
double minfre;
vector<set<string> > datatable;
set<string> products;
map<string, int> freq;

vector<string> wordsof(string str)
{
    vector<string> tmpset;
    string tmp = "";
    int i = 0;
    while (str[i])
    {
        if (isalnum(str[i]))
            tmp += str[i];
        else
        {
            if (tmp.size() > 0)
                tmpset.push_back(tmp);
            tmp = "";
        }
        i++;
    }
    if (tmp.size() > 0)
        tmpset.push_back(tmp);
    return tmpset;
}

string combine(vector<string> &arr, int miss)
{
    string str;
    for (int i = 0; i < arr.size(); i++)
        if (i != miss)
            str += arr[i] + " ";
    str = str.substr(0, str.size() - 1);
    return str;
}
set<string> cloneit(set<string> &arr)
{
    set<string> dup;
    for (set<string>::iterator it = arr.begin(); it != arr.end(); it++)
        dup.insert(*it);
    return dup;
}
set<string> apriori_gen(set<string> &sets, int k)
{
    set<string> set2;
    for (set<string>::iterator it1 = sets.begin(); it1 != sets.end(); it1++)
    {
        set<string>::iterator it2 = it1;
        it2++;
        for (; it2 != sets.end(); it2++)
        {
            vector<string> v1 = wordsof(*it1);
            vector<string> v2 = wordsof(*it2);

            bool alleq = true;
            for (int i = 0; i < k - 1 && alleq; i++)
                if (v1[i] != v2[i])
                    alleq = false;
            if (!alleq)
                continue;

            v1.push_back(v2[k - 1]);
            if (v1[v1.size() - 1] < v1[v1.size() - 2])
                swap(v1[v1.size() - 1], v1[v1.size() - 2]);

            for (int i = 0; i < v1.size() && alleq; i++)
            {
                string tmp = combine(v1, i);
                if (sets.find(tmp) == sets.end())
                    alleq = false;
            }

            if (alleq)
                set2.insert(combine(v1, -1));
        }
    }
    return set2;
}

int main()
{
    fin.open("apriori.in");
    cout << "Frequency % :";
    cin >> minfre;
    string str;
    while (!fin.eof())
    {
        getline(fin, str);
        vector<string> arr = wordsof(str); //taking data from file ,
        set<string> tmpset;
        for (int i = 0; i < arr.size(); i++)
            tmpset.insert(arr[i]);
        datatable.push_back(tmpset);
        for (set<string>::iterator it = tmpset.begin(); it != tmpset.end(); it++)
        {
            products.insert(*it);
            freq[*it]++;
        }
    }
    fin.close();

    cout << "No of transactions: " << datatable.size() << endl;
    minfre = minfre * datatable.size() / 100;
    cout << "Min frequency:" << minfre << endl;
    queue<set<string>::iterator> q;

    for (set<string>::iterator it = products.begin(); it != products.end(); it++)
        if (freq[*it] < minfre)
            q.push(it);
    while (q.size() > 0)
    {
        products.erase(*q.front());
        q.pop();
    }

    for (set<string>::iterator it = products.begin(); it != products.end(); it++)
        cout << *it << " " << freq[*it] << endl;

    int i = 2;
    set<string> prev = cloneit(products);
    while (i)
    {
        set<string> cur = apriori_gen(prev, i - 1);
        if (cur.size() < 1)
            break;
        for (set<string>::iterator it = cur.begin(); it != cur.end(); it++)
        {

            vector<string> arr = wordsof(*it);
            int tot = 0;

            for (int j = 0; j < datatable.size(); j++)
            {
                bool pres = true;
                for (int k = 0; k < arr.size() && pres; k++)
                    if (datatable[j].find(arr[k]) == datatable[j].end())
                        pres = false;
                if (pres)
                    tot++;
            }

            if (tot >= minfre)
                freq[*it] += tot;
            else
                q.push(it);
        }
        while (q.size() > 0)
        {
            cur.erase(*q.front());
            q.pop();
        }

        for (set<string>::iterator it = cur.begin(); it != cur.end(); it++)
            cout << *it << " " << freq[*it] << endl;

        prev = cloneit(cur);
        i++;
    }
}
