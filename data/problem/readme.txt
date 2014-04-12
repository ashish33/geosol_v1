README for GeoSol Database Structure

Author: Min Joon Seo, University of Washington
Last updated: 04/04/2014

This text file describes the database structure for GeoSol.
Its database is primarily based on .xml format, and
it also uses .txt and .png for plain text and image files, respectively.

Each of folders 0000~9999 contains all necessary files for each problem.
Below describes the content of each folder.

-------------------------------------------------------------------------
| about.xml
-------------------------------------------------------------------------
This file contains the background information of the problem.

<src>: The name of the website that the problem was extracted from
<url>: The url of the exact webpage
<level>: The grade level the problem is made for. 6-12, SAT, etc. 
<type>: The type of the question. MC (multiple choice), 


-------------------------------------------------------------------------
| raw_problem.xml
-------------------------------------------------------------------------
This is the raw markup of the problem. (not used for now)


-------------------------------------------------------------------------
| simple_problem.xml
-------------------------------------------------------------------------
This file is a processed version of raw_problem.xml, either manually or 
auto-generated. It assumes that the question is pure ASCII and
the problem has only one diagram. If the problem is a multiple choice,
each choice is only consisted of ASCII text.

<text>: the question text. Must be pure ASCII
<image_ref>:
<mc>: multiple choice.


-------------------------------------------------------------------------
| 000.png~999.png
-------------------------------------------------------------------------
These images will be referenced by 'image_ref.xml'.



-------------------------------------------------------------------------
| image.xml
-------------------------------------------------------------------------
This file contains reference information for images. It tells which image
is the original diagram, which image is the segmented main geometry, and
which images are labels.

<image id=XXX.png>: bracket for each image
    <type>: original,major,minor
    <label>: if applicable
    <loc>: "x,y" for major and minor images
    <size>: "xsize,ysize"


-------------------------------------------------------------------------
| visual_primitive.xml
-------------------------------------------------------------------------
This file contains the visual primitives (lines and circles) obtained
via Hough transform.

<vp id=XXXX>: visual primitive; id is unique to the primitive.
    <type>: 'line' or 'arc'
    <data>: for line, 'x0,y0,x1,y1'. for circle, 'x,y,r,t0,t1'
            if it is a full circle, t0 = t1 = 0. Always clockwise. 

OR in csv format

each line is id,type,data:
0000,line,x0,y0,x1,y1
0001,arc,x,y,r,t0,t1


-------------------------------------------------------------------------
| optimized_vp.xml
-------------------------------------------------------------------------
visual primitives selected via optimization, referenced by id.
<vpid>: XXXX


-------------------------------------------------------------------------
| vertex.xml
-------------------------------------------------------------------------
Interesting points in the diagram.

<vertex id=XXXX>: each point
    <loc>: "x,y"
    <label>: string reference to the point



-------------------------------------------------------------------------
| quantized_vp.xml
-------------------------------------------------------------------------
<qvp id=XXXX>:
    <type>
    <data>
    <vpid>: the original visual primitive this this qvp is coming from
    <family>: qvpid1,qvpid2,qvpid3



-------------------------------------------------------------------------
| vp_graph.xml
-------------------------------------------------------------------------
visual primitive graph constructed by (V,E) where V comes from vertex.xml
and E comes from quantized_vp.xml

<edge id=XXXX>point_id1,point_id2</edge>: edge id is equivalent to 
qvp id

<nbr id=XXXX>qvp_id1,qvp_id2,qvp_id3,...</nbr>: neighbor qvp of a vertex
the nbr id is equivalent to the vertex id


-------------------------------------------------------------------------
| visual_element.xml
-------------------------------------------------------------------------
Each visual element is made of one or more visual primitives from
selected_vp.csv.

<ve id=XXXX>: visual element; id is unique to the element
    <type>: 'line', 'arc', 'circle', 'triangle', 'rectangle', etc.
    <lines>: 'XXXX,
