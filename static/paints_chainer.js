$(function() {
        image_id = "test_id"

        $('#wPaint').wPaint({
            path: '/static/wPaint/',
            menuOffsetLeft: -35,
            menuOffsetTop: -50,
        });

        $("#painting_label").hide();
        $("#submit").prop("disabled",true);

        $("#submit").click( function(){
            if( !$("#background").attr("src") ){
                alert( "select a file" );
            } else {
              colorize();
            }
        });

        $('#img_pane').hide()



        //$('#line_form').on('change', 'input[type="file"]', function(e) {
        $('#load_line_file').on('change', function(e) {
          var file = e.target.files[0],
              reader = new FileReader(),
              $preview = $(".preview");

          if(file.type.indexOf("image") < 0){
            return false;
          }

          reader.onload = (function(file) {
            console.log("up")
            return function(e) { select_src( e.target.result ); }
          })(file);

          reader.readAsDataURL(file);
        });


        $("#set_line_url").click( function(){
          //currently not work 
          select_src( $("#load_line_url").val() )
        });

        
        $('#output').bind('load', function(){
          $('#output')
            .height( $('#background').height() )
            .width( $('#background').width() )
          $('#img_pane')
            .width( $('#output').width()*2.3+24 )
            .height( $('#output').height()+20 )
        });


//--- functions 

        function uniqueid(){
            var idstr=String.fromCharCode(Math.floor((Math.random()*25)+65));
            do {                
                var ascicode=Math.floor((Math.random()*42)+48);
                if (ascicode<58 || ascicode>64){
                    idstr+=String.fromCharCode(ascicode);    
                }                
            } while (idstr.length<32);
            return (idstr);
        } 

        startPaint = function(){
            $("#painting_label").show();
            $("#submit").prop("disabled",true);
            console.log("coloring start");
        }
        endPaint = function(){
            $("#painting_label").hide();
            $("#submit").prop("disabled",false);
            console.log("coloring finish");
        }

        colorize = function(){
            startPaint()
            var ajaxData = new FormData();
            ajaxData.append('line', $("#background").attr("src") );
            ajaxData.append('ref', $("#wPaint").wPaint("image") );
            ajaxData.append('blur', $("#blur_k").val() );
            ajaxData.append('id', image_id );
            $.ajax({
                url: "/post",
                data: ajaxData,
                cache: false,
                contentType: false,
                processData: false,
                type: 'POST',
                dataType:'json',
                complete: function(data) {
                        //location.reload();
                        console.log("uploaded")
                        var now = new Date().getTime();
                        $('#output').attr('src', '/static/images/out/'+image_id+'_0.jpg?' + now);
                        $('#output_min').attr('src', '/static/images/out_min/'+image_id+'_0.png?' + now);
					    $('#output_links').html( 
							"<h4>ダウンロード</h4>"
							+ "<a href='/static/images/out/" + image_id + "tmp_out.png' target='_blank'>元サイズに拡大</a><br>"
							+ "<a href='/static/images/out/" + image_id + "med_mult.png' target='_blank'>元サイズで線画と色を合成</a><br>"
							+ "<a href='/static/images/out/" + image_id + "median_color.png' target='_blank'>元サイズで色のみ（線画部分は白色）</a><br>"
                			+ "<a href='/static/images/out/" + image_id + "median_color_soft.png' target='_blank'>元サイズで色のみ ソフト（線画部分は白色）</a><br>"
							);
                        endPaint()
                }
              });
        }

        enable_interactive = function(){
          $('.wPaint-canvas').mouseup(colorize)
        }

        select_src = function(src){
          $("#img_pane").show(
            "fast", function(){
              image_id = uniqueid()

              $("#background").attr("src", src );
              $("#wPaint")
                 .width($("#background").width())
                 .height($("#background").height());

              $("#wPaint").wPaint('resize');
              $("#submit").prop("disabled",true)
              colorize();
          });
        };



});
