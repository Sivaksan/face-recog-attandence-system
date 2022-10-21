<?php
include 'config.php';

// if($conn)
// {
//     echo "Connection Successfull";
// }
// else {
//     echo "Connection Failed";
// }
$qry="SELECT * FROM attendance";
$connection = mysqli_query($conn, $qry);
$data = mysqli_fetch_assoc($connection);
$num=mysqli_num_rows($connection);

?>


<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" href="nav.css">

    <title>Attendace</title>
  </head>
  <body>
    

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.7/dist/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>

    <div class="nav">
        <input type="checkbox" id="nav-check">
        <div class="nav-header">
          <div class="nav-title">
           <a href="index.php"> Face Recognition App</a>
          </div>
        </div>
        <div class="nav-btn">
          <label for="nav-check">
            <span></span>
            <span></span>
            <span></span>
          </label>
        </div>
        
        <div class="nav-links">
           
          <a href="signin.php">Admin</a>
          <a href="panalselector.php" >Panal Selector</a>
        </div>
      </div>
<div>

<div class="container">
        <br>
        <div class="row">
           <div class="col-md-8 mx-auto">
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
                             <h4>Student Attendace</h4>
                            </center>
                       </div>
                    </div>
                    <div class="row">
                       <div class="col">
                          <hr>
                       </div>
                    </div>

                    <table class="table table-dark">
  <thead>
    <tr>
      
      <th scope="col">Name</th>
      <th scope="col">Time</th>
      
    </tr>
  </thead>
<?php
if($num>0)
{
    while($data =mysqli_fetch_assoc($connection))
    {
       echo "
       <tr>
       <td>".$data['Name']."</td>
       <td>".$data['Time']."</td>
       </tr>
       ";

       
    }
}
?>
  </tbody>
</table>   
                   
</div>
</form>
</body>
</html>