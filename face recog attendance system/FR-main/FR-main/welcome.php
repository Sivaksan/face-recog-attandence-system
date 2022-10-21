<?php 
session_start();
?>

<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <title>Create Account</title>
<style>
body {
    background: #edc0bf;
    background: linear-gradient(90deg, #edc0bf 0,#c4caef 58%);
    
}


  </style>

  </head>
  <body>
    <center>
    <h1 class="display-4">WELCOME ADMIN</h1>
</center>
    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.7/dist/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
    <!you--datatables.css-->
    <link href="datatables/css/jquery.dataTables.min.css" rel="stylesheet" />
<center>
    <a href="logout.php"><button type="button" class="btn btn-outline-danger">LOG-OUT</button></a>
    <a href="attendance.php"><button type="button" class="btn btn-outline-warning">View Attendance</button></a>
</center>
<br />






 
    <div class="container">
      
      <?php 
if(isset($_SESSION['status']) && $_SESSION!=='')
{
  ?>
  <div class="alert alert-warning alert-dismissible fade show" role="alert">
  <strong>HELLO</strong><?php echo $_SESSION['status']; ?>
  <button type="button" class="close" data-dismiss="alert" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</div>
<?php
  unset($_SESSION['status']);
}
      ?>
   
    <form method="post" enctype="multipart/form-data" action="studcreate.php">
      <div class="row">
         <div class="col-9">
            <div class="card">
               <div class="card-body">
                  <div class="row">
                     <div class="col">
                        <center>
                           <img width="100px" src="images/user.png"/>
                        </center>
                     </div>
                  </div>
                  <div class="row">
                     <div class="col">
                        <center>
                           <h4>CREATE STUDENT PROFILE</h4>

                            
                          </center>
                     </div>
                  </div>
                  <div class="row">
                     <div class="col">
                        <hr>
                     </div>
                  </div>
                  <div class="row">
                     <div class="col-md-6">
                        <label>Full Name</label>
                        <div class="form-group">
                           <input type="text" class="form-control" placeholder="Full Name" name="fullname" required>
                        </div>
                     </div>
                     <div class="col-md-6">
                        <label>Index Number</label>
                        <div class="form-group">
                        <input type="number" class="form-control"  placeholder="Index Number" name="index" required>
                        </div>
                     </div>
                  </div>



                  <div class="row">
                     <div class="col">
                         </div>
                  </div>
                  <div class="row">
                     <div class="col-md-6">
                        <label>E-Mail</label>
                        <div class="form-group">
                        <input type="email" class="form-control" placeholder="Email" name="email" required>
                        </div>
                     </div>
                     <div class="col-md-6">
                        
                        <div class="form-group">
                        <label>Degree</label>
                        <div class="form-group">
                        <select class="custom-select" name="degree">
  <option selected value="se">Software Engineering</option>
  <option value="cs">Computer Science</option>
  <option value="cn">Computer Network</option>
  <option value="csec">Computer Security</option>
</select>
                      
                        </div>
                        </div>
                     </div>
</div>                              
<br />
                  <div class="row">
                     <div class="col">
                       <input type="file" class="form-control" name="imageupload" id="imageupload" required>
                     </div>
                     
                     
                  </div>


                  
                  </div>

                  


                  <div class="row">
                     <div class="col-md-6">
                        <label>User ID</label>
                        <div class="form-group">
                           <input type="number" class="form-control" ID="TextBox8"  placeholder="User ID" name="userid" required >
                        </div>
                     </div>
                     

                       <div class="col-md-5">
                        <label>New Password</label>
                        <div class="form-group">
                           <input type="password"class="form-control"  placeholder="New Password" TextMode="Password" name="password" required>
                        </div>
                     </div>
                    
                  </div>
                  <div class="row">
                     <div class="col-8 mx-auto">
                        <center>
                            <br />
                           <div class="form-group">
                              <input type="submit" class="btn btn-primary" ID="Button1" value="Create Account" name="submit"/>

                              </form>
                           </div>
                           </center>
                         
                     </div>
                    
                  </div>
               </div>
               <a href="index.php"><< Back to Home</a>
            </div>
           
            
            </div>
       
</body>
</html>




