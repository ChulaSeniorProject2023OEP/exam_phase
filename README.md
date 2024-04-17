# AI - Online Proctoring System

Online Proctoring System using video data for behavior analysis. This project aims to create an automated proctoring system where the user can be monitored automatically through the webcam.

<hr/>

🔗 For more information : [OEP Project](https://sarmkunatham.notion.site/Online-Proctoring-System-0f1eaff984634a128e3c3c6051811d24)

utomated Online Exam Proctoring
Yousef Atoum, Liping Chen, Alex X. Liu, Stephen D. H. Hsu, and Xiaoming Liu
IEEE TRANSACTION ON MULTIMEDIA
-------------------------------------------------------------------------------

A total of 24 subjects, all of whom are students at Michigan State University, participated in the data collection. 15 subjects were actors that pretended to be taking the exam. They were asked to perform cheating behaviors during the session, without any instructions on what cheating behavior to perform or how to perform them. One issue with these subjects is that potentially artificial behaviors are observed during the acting. Therefore, to capture real-world exam scenarios, we asked nine students to take the real exam, where their scores were recorded. Knowing that they are not likely to cheat in the data capturing room, the proctor invokes the cheating behaviors by talking, walking up to the student, or handing them a book, etc. The combination of these two types of subjects enriches the database with various cheat techniques, as well as the sense of engagement in real exams.

In our database, the acting student videos and data are located in folders:
subject1
subject2
subject3
subject4
subject5
subject6
subject7
subject8
subject9
subject17
subject20
subject21
subject22
subject23
subject24

The nine real student videos and data are located in folders:
subject10
subject11
subject12
subject13
subject14
subject15
subject16
subject18
subject19

In each folder, you will find 4 files:
1- gt.txt : Groundtruth file. The labeling of one cheat instance consists of three 	pieces of information (a) the start time, (b) end time and (c) type of cheating.
2- Audio file (.wav) : This file contains the audio information of the test taking. Note that the test taker is required to enter a username at the beginning of a test session. Therefore, we use the username as the name of the audio (wav) file. 
3- webcam video (.avi): Similar to the audio, this file will contain the username of the test taker followed by "1" (For example "yousef1.avi" in subject1 folder). This video was captured using the webcam located above the monitor.
4- wearcam video (.avi): Similar to the audio, this file will contain the username of the test taker followed by "2" (For example "yousef2.avi" in subject1 folder). This video was captured using the wearcam attached to the eyeglasses.

## Detect suspicious behavior of examinee

Capturing video and generate alert report on 3 aspects:
- 1. examinee head pose estimation
    - 1. normal orientation
    - 2. head away
- 2. examinee iris tracking
    - 1. looking at the screen
    - 2. looking away
- 3. examinee lip movement
    - 1. lip close
    - 2. lip open
 