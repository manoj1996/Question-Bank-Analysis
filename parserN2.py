import os
import xml.dom.minidom

#converts a list of page numbers to a comma-separated string for pdf2txt.py -p 
def list2string(lyst):
 k = 0
 string = ""
 while k < len(lyst):
  if(k == len(lyst) - 1):
   string = string + str(lyst[k])
  else:
   string = string + str(lyst[k]) + ", "
  k += 1
 return string

#returns the value in the <pageno> tag 
def page_extractor(index):
 lyst = []
 num = 0
 for i in n[index].childNodes[3].toxml():
  if i.isdigit():
   lyst.append(int(i))
 
 for i in range(len(lyst)):
  num += lyst[i]*(10**(len(lyst)-i-1))
  
 return num
 
os.system("dumppdf.py -T'/home/manoj/Documents/N^2Algo/[William_Navidi]_Statistics_for_Engineers_and_Scie(Book4You).pdf' > out.xml")
doc = xml.dom.minidom.parse("out.xml");
n = doc.getElementsByTagName("outline")

#accept first and last chapter title
chap1 = raw_input("Enter the name of the first chapter.")
chap2 = raw_input("Enter the name of the last chapter.")

#find first and last chapter indices, count1 and count2
i = 0
while(i < n.length):
 if n[i].getAttribute('title') == chap1:
  count1 = i
  break
 i += 1
 
while(i < n.length):
 if n[i].getAttribute('title') == chap2:
  count2 = i
  break
 i += 1
 
print(count1)
print(count2)

#put the text of each topic into a separate file
ctr = 0
for i in range(count1, count2):
 pointer1 = n[i] #points to a chapter
 add = i+1
 pointer2 = n[add] #points to the chapter immediately after it
 while int(pointer2.getAttribute('level')) > 3: #will it compare the string to the number?
  add += 1
  pointer2 = n[add]
 pglist = [x for x in range(page_extractor(i), page_extractor(add))]
 #print "pdf2txt.py -p " + '"' + list2string(pglist) + '"' + " -o file{0}.txt".format(ctr+1) + " /home/karan/Textbooks/\[William_Navidi\]_Statistics_for_Engineers_and_Scie\(Book4You\).pdf"
 os.system("pdf2txt.py -p " + '"' + list2string(pglist) + '"' + " -o file{0}.txt".format(ctr+1) + " /home/karan/Textbooks/\[William_Navidi\]_Statistics_for_Engineers_and_Scie\(Book4You\).pdf")
 i = add 
 ctr += 1

