<?php 
include 'config.php';

session_start();


if (isset($_POST['submit'])) {
	$fullname = $_POST['fullname'];
	$index = $_POST['index'];
  $email = $_POST['email'];
  $degree = $_POST['degree'];
  $imageupload = $_FILES['imageupload']['name'];
  $userid = $_POST['userid'];
  $password = md5($_POST['password']);
	
$qry="INSERT INTO student (fullname,indexnum,email,degree,imageupload,userid,pwd) VALUES ('$fullname','$index','$email','$degree','$imageupload','$userid','$password')";

$qry_run = mysqli_query($conn,$qry);

if($qry_run)
{
    move_uploaded_file($_FILES["imageupload"]["tmp_name"],"upload/".$_FILES["imageupload"]["name"]);
    $_SESSION['status'] ="Account created Successfully";
    header("Location:welcome.php");
}
else
{
    $_SESSION['status'] ="Error Uploading Image";
    header("Location:studcreate.php");
}


	
}



?>